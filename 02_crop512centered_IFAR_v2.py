#!/usr/bin/env python3
"""
02_crop512centered_IFAR_v2.py

Purpose
-------
Given a CSV that contains (at least) the original image filename and an approximate pollen center (x, y),
crop a fixed-size square (default 512x512) from the *original* image so that (x, y) lands near the center.

This script is designed to be the second step in an IFAR-reproducible pipeline:

  01_round_detect_400_dbscan_IFAR_v2.py  -> produces circle candidates (original_filename, x, y, r, cropped_filename)
  (manual)                              -> you may add 'type' column to the CSV after visual inspection
  02_crop512centered_IFAR_v2.py          -> crops exact 512x512 tiles from the original images
  03_estimate_center_cxcy_IFAR.py        -> estimates cx, cy from the 512x512 tiles (mask/contour based)
  04_make_features10_IFAR.py             -> makes 10 features + *_zscore

Key DOI/reproducibility improvements vs older scripts
-----------------------------------------------------
- CLI arguments (no hard-coded paths)
- Always writes a per-row status, so missing images never shift alignment
- Uses per-row assignment (no list/zip mismatch bugs)
- Ensures the output crop is exactly size x size via black padding if near borders

Example
-------
python 02_crop512centered_IFAR_v2.py \
  --csv_path out/circle_data_with_type.csv \
  --image_dir pollen_picture_folder \
  --output_dir out/cropped_512 \
  --size 512
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("crop512")


def crop_centered_square(image: np.ndarray, center_x: int, center_y: int, size: int = 512) -> np.ndarray:
    """
    Crop a square region of shape (size, size, 3) from image so that (center_x, center_y) is at its center.
    If the crop extends outside the image boundary, pad the missing part with black pixels.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C).
    center_x, center_y : int
        Center coordinates in the input image coordinate system.
    size : int
        Output crop size (default 512).

    Returns
    -------
    np.ndarray
        Cropped image of shape (size, size, 3).
    """
    if image is None:
        raise ValueError("image is None")

    h, w = image.shape[:2]
    half = size // 2

    x_start = center_x - half
    y_start = center_y - half
    x_end = x_start + size
    y_end = y_start + size

    # Output canvas (black)
    result = np.zeros((size, size, 3), dtype=np.uint8)

    # Source bounds clipped to image
    src_x0 = max(0, x_start)
    src_y0 = max(0, y_start)
    src_x1 = min(w, x_end)
    src_y1 = min(h, y_end)

    # Destination bounds in result
    dst_x0 = src_x0 - x_start
    dst_y0 = src_y0 - y_start
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    # Copy
    if (src_x1 > src_x0) and (src_y1 > src_y0):
        result[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]

    return result


def build_suffix_from_folder(folder: Path) -> str:
    """
    Create a stable suffix from folder name.
    Old code used: "_".join(folder_name.split("_")[-2:])
    We'll keep that behavior but fall back safely.
    """
    parts = folder.name.split("_")
    if len(parts) >= 2:
        return "_".join(parts[-2:])
    return folder.name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crop exact 512x512 (or size x size) patches centered at (x,y).")

    p.add_argument("--csv_path", type=Path, required=True, help="Input CSV with original_filename and (x,y).")
    p.add_argument("--image_dir", type=Path, required=True, help="Folder containing the original images.")
    p.add_argument("--output_dir", type=Path, required=True, help="Output folder for cropped images and CSV log.")

    p.add_argument("--size", type=int, default=512, help="Crop size (default 512).")
    p.add_argument("--x_col", type=str, default="x", help="Column name for x coordinate (default: x).")
    p.add_argument("--y_col", type=str, default="y", help="Column name for y coordinate (default: y).")
    p.add_argument("--file_col", type=str, default="original_filename", help="Column name for original filename.")
    p.add_argument("--type_col", type=str, default="type", help="Column name for type label (optional).")

    p.add_argument(
        "--name_mode",
        choices=["type_index", "original_index"],
        default="type_index",
        help=(
            "Output filename scheme. "
            "'type_index': <type>_<suffix>_centered_<rowid>.jpg (old behavior). "
            "'original_index': <original_stem>_centered_<rowid>.jpg (more general)."
        ),
    )

    p.add_argument(
        "--out_csv",
        type=Path,
        default=None,
        help="Output CSV path. If omitted, will be created inside output_dir with a timestamp.",
    )

    p.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")

    if args.size <= 0 or args.size % 2 != 0:
        raise ValueError("--size must be a positive even integer (e.g., 512).")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {args.image_dir}")

    suffix = build_suffix_from_folder(args.image_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")

    out_csv = args.out_csv or (output_dir / f"crop512_log_{suffix}_{timestamp}.csv")

    df = pd.read_csv(args.csv_path)

    required = {args.file_col, args.x_col, args.y_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Prepare output columns (safe per-row writes)
    if "saved_filename" not in df.columns:
        df["saved_filename"] = pd.NA
    if "crop_status" not in df.columns:
        df["crop_status"] = pd.NA

    total = len(df)
    saved = 0
    skipped = 0

    for row_id, row in df.iterrows():
        try:
            original_filename = str(row[args.file_col])
            x = int(row[args.x_col])
            y = int(row[args.y_col])
        except Exception as e:
            df.at[row_id, "crop_status"] = f"bad_row:{type(e).__name__}"
            skipped += 1
            LOGGER.warning("Row %s: cannot parse filename/x/y (%s)", row_id, e)
            continue

        image_path = args.image_dir / original_filename
        if not image_path.exists():
            df.at[row_id, "crop_status"] = "missing_image"
            skipped += 1
            LOGGER.warning("Row %s: image not found: %s", row_id, image_path)
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            df.at[row_id, "crop_status"] = "read_failed"
            skipped += 1
            LOGGER.warning("Row %s: cv2.imread failed: %s", row_id, image_path)
            continue

        cropped = crop_centered_square(image, x, y, size=args.size)

        # Build output filename
        if args.name_mode == "type_index":
            type_label = str(row[args.type_col]) if args.type_col in df.columns else "unknown"
            safe_type = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in type_label)
            save_name = f"{safe_type}_{suffix}_centered_{int(row_id):04d}.jpg"
        else:
            stem = image_path.stem
            save_name = f"{stem}_centered_{int(row_id):04d}.jpg"

        save_path = output_dir / save_name
        ok = cv2.imwrite(str(save_path), cropped)
        if not ok:
            df.at[row_id, "crop_status"] = "write_failed"
            skipped += 1
            LOGGER.error("Row %s: cv2.imwrite failed: %s", row_id, save_path)
            continue

        df.at[row_id, "saved_filename"] = save_name
        df.at[row_id, "crop_status"] = "ok"
        saved += 1

        if saved % 100 == 0:
            LOGGER.info("Progress: %d/%d saved", saved, total)

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    LOGGER.info("Done. saved=%d, skipped=%d, total=%d", saved, skipped, total)
    LOGGER.info("CSV written: %s", out_csv)
    LOGGER.info("Crops folder: %s", output_dir)


if __name__ == "__main__":
    main()
