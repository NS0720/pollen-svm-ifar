#!/usr/bin/env python3
"""
03_estimate_center_cxcy_IFAR_v3.py

Purpose
-------
Estimate pollen center coordinates (cx, cy) and internal intensity-offset metrics from 512x512 crops.

This script is designed to follow:
  02_crop512centered_IFAR_v2.py  -> produces 'saved_filename' and (optionally) 'crop_status'

and to precede:
  04_make_features10_IFAR.py     -> consumes cx, cy (and other columns) for feature engineering.

Key robustness improvements (DOI-friendly)
------------------------------------------
- No hard-coded paths: fully CLI-driven.
- Never misaligns rows: writes results per-row; keeps original row order.
- Can skip non-ok rows if 'crop_status' exists (default).
- Adds 'center_status' describing per-row outcomes.
- Optional debug overlays saved to disk (no GUI / no plt.show).

Outputs
-------
A CSV identical to the input plus these columns:
  cx, cy, gx, gy, dist, offset_ratio, center_status

Notes
-----
- (cx, cy) is obtained from the maximum contour (minEnclosingCircle).
- (gx, gy) is the centroid of pixels above a percentile threshold, limited to a radius fraction.

"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def setup_logger(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("03_estimate_center_cxcy_IFAR_v2")


def estimate_center_and_offset(
    img_gray: np.ndarray,
    blur: bool = True,
    high_percentile: float = 80.0,
    max_radius_fraction: float = 0.8,
) -> Tuple[Optional[Tuple[int, int, float, float, float, float]], str]:
    """
    Returns:
        (cx, cy, gx, gy, dist, offset_ratio), status
    """
    if img_gray is None:
        return None, "read_failed"

    if blur:
        img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Extract contours through binarization
    try:
        _, thresh = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        contours, _hier = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
    except Exception:
        return None, "threshold_or_contours_failed"

    if not contours:
        return None, "no_contour"

    cnt = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    cx_i, cy_i, r_i = int(cx), int(cy), int(r)

    if r_i <= 0:
        return None, "bad_radius"

    # "High" intensity threshold (top 20%)
    t_high = np.percentile(img_gray, high_percentile)
    mask = img_gray > t_high
    coords = np.column_stack(np.where(mask))  # (row=y, col=x)

    if coords.size == 0:
        # Still return contour-based center; intensity-based offset cannot be computed
        return (cx_i, cy_i, float("nan"), float("nan"), float("nan"), float("nan")), "ok_partial_no_pixels_over_threshold"

    # Limit to max_radius_fraction * r from the pollen center
    distances = np.linalg.norm(coords - np.array([cy_i, cx_i]), axis=1)  # y,x order
    coords_filtered = coords[distances <= (max_radius_fraction * r_i)]

    if len(coords_filtered) == 0:
        # Still return contour-based center; intensity-based offset cannot be computed
        return (cx_i, cy_i, float("nan"), float("nan"), float("nan"), float("nan")), "ok_partial_no_pixels_in_radius"

    gy, gx = coords_filtered.mean(axis=0)  # y, x
    dist = float(np.linalg.norm([gx - cx_i, gy - cy_i]))
    offset_ratio = float(dist / r_i)

    return (cx_i, cy_i, float(gx), float(gy), dist, offset_ratio), "ok"


def save_debug_overlay(
    img_gray: np.ndarray,
    out_path: Path,
    cx: int,
    cy: int,
    gx: float,
    gy: float,
    r_for_circle: Optional[int] = None,
) -> None:
    """Save a simple overlay image for debugging."""
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    if r_for_circle is not None and r_for_circle > 0:
        cv2.circle(img_color, (cx, cy), int(r_for_circle), (255, 0, 0), 1)

    cv2.circle(img_color, (cx, cy), 3, (0, 255, 255), -1)  # center
    cv2.circle(img_color, (int(gx), int(gy)), 3, (0, 0, 255), -1)  # gravity

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img_color)


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate cx,cy and offset features from 512x512 crops.")
    parser.add_argument("--csv_path", required=True, help="CSV produced by step 02 (must include saved_filename).")
    parser.add_argument("--image_dir", required=True, help="Directory containing the 512x512 crop images.")
    parser.add_argument("--output_csv", default=None, help="Output CSV path. Default: alongside input with suffix _center.csv")
    parser.add_argument("--filter_crop_ok", action="store_true", help="If crop_status exists, process only rows where crop_status=='ok'.")
    parser.add_argument("--no_blur", action="store_true", help="Disable Gaussian blur before processing.")
    parser.add_argument("--high_percentile", type=float, default=80.0, help="Percentile threshold for intensity mask (default: 80).")
    parser.add_argument("--max_radius_fraction", type=float, default=0.8, help="Keep pixels within this fraction of r (default: 0.8).")
    parser.add_argument("--debug_overlay_dir", default=None, help="If set, save overlay images here for rows with status ok.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    logger = setup_logger(args.verbose)

    csv_path = Path(args.csv_path)
    image_dir = Path(args.image_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    df = pd.read_csv(csv_path)

    if "saved_filename" not in df.columns:
        raise ValueError("Input CSV must contain 'saved_filename' column from step 02.")

    # Prepare output columns (ensure they exist)
    for col in ["cx", "cy", "gx", "gy", "dist", "offset_ratio", "center_status"]:
        if col not in df.columns:
            df[col] = np.nan if col != "center_status" else ""

    blur = not args.no_blur

    # Processing
    n_ok = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Optional filtering by crop_status
        if args.filter_crop_ok and "crop_status" in df.columns:
            if str(row.get("crop_status", "")).strip().lower() != "ok":
                df.at[idx, "center_status"] = "skipped_not_ok_crop"
                continue

        saved = row.get("saved_filename", "")
        if not isinstance(saved, str) or saved.strip() == "" or saved.lower() == "nan":
            df.at[idx, "center_status"] = "missing_saved_filename"
            continue

        img_path = image_dir / saved
        if not img_path.exists():
            df.at[idx, "center_status"] = "missing_image"
            continue

        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        result, status = estimate_center_and_offset(
            img_gray,
            blur=blur,
            high_percentile=args.high_percentile,
            max_radius_fraction=args.max_radius_fraction,
        )

        if result is None:
            df.at[idx, "center_status"] = status
            continue

        cx, cy, gx, gy, dist, offset_ratio = result
        df.at[idx, "cx"] = cx
        df.at[idx, "cy"] = cy
        df.at[idx, "gx"] = gx
        df.at[idx, "gy"] = gy
        df.at[idx, "dist"] = dist
        df.at[idx, "offset_ratio"] = offset_ratio
        df.at[idx, "center_status"] = "ok"
        n_ok += 1

        if args.debug_overlay_dir:
            # Recompute r for a circle overlay (optional, for visual sanity check)
            # Use the same contour logic quickly:
            try:
                _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt = max(contours, key=cv2.contourArea) if contours else None
                r_circle = int(cv2.minEnclosingCircle(cnt)[1]) if cnt is not None else None
            except Exception:
                r_circle = None

            out_overlay = Path(args.debug_overlay_dir) / (Path(saved).stem + "_overlay.jpg")
            save_debug_overlay(img_gray, out_overlay, cx, cy, gx, gy, r_circle)

    # Output CSV path
    if args.output_csv:
        out_csv = Path(args.output_csv)
    else:
        out_csv = csv_path.with_name(csv_path.stem + "_center.csv")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info("Done. center_status startswith 'ok' rows: %d / %d", n_ok, len(df))
    logger.info("Saved: %s", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
