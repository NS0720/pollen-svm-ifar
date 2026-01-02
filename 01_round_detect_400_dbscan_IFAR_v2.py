"""
round_detect_400_dbscan_IFAR.py

Detect (approximately) circular objects (e.g., pollen grains) in 400x bright-field microscope images
using HoughCircles, merge duplicate detections with DBSCAN, and crop square patches around each
detected object. Optionally logs (x, y, r) and output filenames to CSV.

Author: (add your name)
License: (add a license, e.g., MIT)
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.cluster import DBSCAN


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Circle:
    x: int
    y: int
    r: int


@dataclass(frozen=True)
class CropItem:
    circle: Circle
    crop: np.ndarray


def round_detect(
    image_bgr: np.ndarray,
    *,
    blur_ksize: Tuple[int, int] = (13, 13),
    hough_dp: float = 1.0,
    hough_min_dist: float = 10.0,
    canny_high_threshold: float = 50.0,
    accumulator_threshold: float = 40.0,
    min_radius: int = 50,
    max_radius: int = 350,
) -> np.ndarray:
    """
    Detect circles in a BGR image via HoughCircles.

    Returns:
        circles: ndarray shape (N, 3) with columns [x, y, r], dtype=int
                 Returns an empty array if nothing is detected.
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.empty((0, 3), dtype=int)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, blur_ksize, 0)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=hough_dp,
        minDist=hough_min_dist,
        param1=canny_high_threshold,
        param2=accumulator_threshold,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return np.empty((0, 3), dtype=int)

    circles = np.round(circles[0]).astype(int)
    return circles


def combine_duplicates_dbscan(
    circles: np.ndarray,
    *,
    eps: float = 100.0,
    min_samples: int = 1,
) -> np.ndarray:
    """
    Merge near-duplicate circle detections using DBSCAN on center coordinates.

    Args:
        circles: ndarray shape (N, 3) [x, y, r]
        eps: maximum distance between centers to be clustered as the same object (pixels)
        min_samples: DBSCAN parameter

    Returns:
        merged: ndarray shape (K, 3) [x, y, r]
    """
    if circles is None or len(circles) == 0:
        return np.empty((0, 3), dtype=int)

    circles = np.round(circles).astype(int)

    coords = circles[:, :2]
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_

    merged: List[List[int]] = []
    for label in sorted(set(labels)):
        group = circles[labels == label]
        avg_x = int(np.mean(group[:, 0]))
        avg_y = int(np.mean(group[:, 1]))
        avg_r = int(np.mean(group[:, 2]))
        merged.append([avg_x, avg_y, avg_r])

    return np.array(merged, dtype=int)


def crop_squares(
    circles: np.ndarray,
    image_bgr: np.ndarray,
    *,
    scale: float = 2.0,
) -> List[CropItem]:
    """
    Crop square regions around circles.

    The crop box is:
        x in [x - scale*r, x + scale*r]
        y in [y - scale*r, y + scale*r]

    Returns:
        List of CropItem(circle, crop). Items that would produce an empty crop are skipped.
    """
    if image_bgr is None or image_bgr.size == 0 or circles is None or len(circles) == 0:
        return []

    height, width = image_bgr.shape[:2]
    items: List[CropItem] = []

    for (x, y, r) in circles:
        x, y, r = int(x), int(y), int(r)
        half = int(round(scale * r))

        x0 = max(x - half, 0)
        x1 = min(x + half, width)
        y0 = max(y - half, 0)
        y1 = min(y + half, height)

        if x1 <= x0 or y1 <= y0:
            LOGGER.debug("Skip invalid crop range: (x=%s,y=%s,r=%s)", x, y, r)
            continue

        crop = image_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            LOGGER.debug("Skip blank crop: (x=%s,y=%s,r=%s)", x, y, r)
            continue

        items.append(CropItem(Circle(x, y, r), crop))

    return items


def draw_circles(image_bgr: np.ndarray, circles: np.ndarray) -> np.ndarray:
    """Return a copy of the image with circles drawn (for quick QA)."""
    out = image_bgr.copy()
    for (x, y, r) in circles:
        cv2.circle(out, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.circle(out, (int(x), int(y)), 2, (0, 0, 255), 3)
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_output_dir(base: Optional[Path] = None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    out = (base if base is not None else Path.cwd()) / f"output_{ts}"
    ensure_dir(out)
    return out


def make_crop_filename(
    *,
    original_stem: str,
    idx: int,
    timestamp: str,
    deterministic: bool,
) -> str:
    if deterministic:
        return f"{original_stem}_crop_{idx:03d}.jpg"
    return f"cropped_image_{timestamp}_{idx}.jpg"


def process_one_image(
    image_path: Path,
    output_dir: Path,
    csv_writer: Optional[csv.writer],
    *,
    dbscan_eps: float,
    dbscan_min_samples: int,
    blur_ksize: Tuple[int, int],
    hough_dp: float,
    hough_min_dist: float,
    canny_high_threshold: float,
    accumulator_threshold: float,
    min_radius: int,
    max_radius: int,
    crop_scale: float,
    deterministic_names: bool,
    save_debug_overlay: bool,
) -> int:
    """
    Returns number of crops saved.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        LOGGER.warning("Failed to read image: %s", image_path)
        return 0

    circles = round_detect(
        image,
        blur_ksize=blur_ksize,
        hough_dp=hough_dp,
        hough_min_dist=hough_min_dist,
        canny_high_threshold=canny_high_threshold,
        accumulator_threshold=accumulator_threshold,
        min_radius=min_radius,
        max_radius=max_radius,
    )

    if len(circles) == 0:
        LOGGER.info("No circles detected: %s", image_path.name)
        return 0

    merged = combine_duplicates_dbscan(circles, eps=dbscan_eps, min_samples=dbscan_min_samples)
    items = crop_squares(merged, image, scale=crop_scale)

    if save_debug_overlay:
        overlay = draw_circles(image, merged)
        overlay_path = output_dir / f"{image_path.stem}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    saved = 0
    for i, item in enumerate(items):
        fname = make_crop_filename(
            original_stem=image_path.stem,
            idx=i,
            timestamp=ts,
            deterministic=deterministic_names,
        )
        out_path = output_dir / fname
        ok = cv2.imwrite(str(out_path), item.crop)
        if not ok:
            LOGGER.warning("Failed to write crop: %s", out_path)
            continue

        saved += 1
        if csv_writer is not None:
            csv_writer.writerow([image_path.name, item.circle.x, item.circle.y, item.circle.r, fname])

    LOGGER.info("Saved %d crops from %s", saved, image_path.name)
    return saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect circular objects (400x) using HoughCircles + DBSCAN and crop squares."
    )
    p.add_argument(
        "--input_dir",
        type=Path,
        default=Path("pollen_picture_folder"),
        help="Folder containing .jpg images (default: pollen_picture_folder).",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output folder. If omitted, creates output_<timestamp> in current directory.",
    )
    p.add_argument(
        "--write_csv",
        action="store_true",
        help="Write CSV log (original_filename,x,y,r,cropped_filename) into output_dir.",
    )
    p.add_argument(
        "--csv_name",
        type=str,
        default=None,
        help="CSV filename (default: <input_dir_name>_circle_data.csv).",
    )
    p.add_argument("--dbscan_eps", type=float, default=100.0)
    p.add_argument("--dbscan_min_samples", type=int, default=1)

    # Hough / preprocessing
    p.add_argument("--blur", type=int, default=13, help="GaussianBlur kernel size (odd integer).")
    p.add_argument("--hough_dp", type=float, default=1.0)
    p.add_argument("--hough_min_dist", type=float, default=10.0)
    p.add_argument("--canny_high", type=float, default=50.0)
    p.add_argument("--hough_thresh", type=float, default=40.0)
    p.add_argument("--min_radius", type=int, default=50)
    p.add_argument("--max_radius", type=int, default=350)

    # crop
    p.add_argument("--crop_scale", type=float, default=2.0, help="Crop half-width as scale*r.")
    p.add_argument(
        "--deterministic_names",
        action="store_true",
        help="Use stable filenames: <original>_crop_###.jpg (recommended for reproducibility).",
    )
    p.add_argument(
        "--save_debug_overlay",
        action="store_true",
        help="Save <original>_overlay.jpg with detected circles drawn.",
    )

    p.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    input_dir: Path = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found or not a directory: {input_dir}")

    output_dir: Path
    if args.output_dir is None:
        output_dir = make_output_dir()
    else:
        output_dir = args.output_dir
        ensure_dir(output_dir)

    # deterministic ordering is good practice for research code
    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}])

    csv_writer: Optional[csv.writer] = None
    csv_file = None

    if args.write_csv:
        csv_name = args.csv_name or f"{input_dir.name}_circle_data.csv"
        csv_path = output_dir / csv_name
        csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["original_filename", "x", "y", "r", "cropped_filename"])
        LOGGER.info("Writing CSV: %s", csv_path)

    blur = int(args.blur)
    if blur % 2 == 0:
        raise ValueError("--blur must be an odd integer (e.g., 11, 13, 15).")

    total_saved = 0
    try:
        for image_path in image_paths:
            total_saved += process_one_image(
                image_path,
                output_dir,
                csv_writer,
                dbscan_eps=args.dbscan_eps,
                dbscan_min_samples=args.dbscan_min_samples,
                blur_ksize=(blur, blur),
                hough_dp=args.hough_dp,
                hough_min_dist=args.hough_min_dist,
                canny_high_threshold=args.canny_high,
                accumulator_threshold=args.hough_thresh,
                min_radius=args.min_radius,
                max_radius=args.max_radius,
                crop_scale=args.crop_scale,
                deterministic_names=args.deterministic_names,
                save_debug_overlay=args.save_debug_overlay,
            )
    finally:
        if csv_file is not None:
            csv_file.close()

    LOGGER.info("Done. Total crops saved: %d", total_saved)
    LOGGER.info("Output dir: %s", output_dir)


if __name__ == "__main__":
    main()
