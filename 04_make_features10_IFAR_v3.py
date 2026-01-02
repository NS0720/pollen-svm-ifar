#!/usr/bin/env python3
"""
04_make_features10_IFAR_v3.py

Purpose
-------
Generate 10 engineered features from 512x512 pollen crops and append *_zscore columns.

This script is designed to follow:
  02_crop512centered_IFAR_v2.py  -> produces 'saved_filename' (+ optional 'crop_status')
  03_estimate_center_cxcy_IFAR_v2.py -> produces 'cx','cy' (+ optional 'center_status', 'gx','gy','dist','offset_ratio')

Robustness (DOI-friendly)
-------------------------
- Never misalign rows: writes feature columns per-row (df.at[idx, ...]).
- Optional filtering: process only rows where crop_status=='ok' and/or center_status=='ok'.
- Records per-row processing state in 'feature_status'.
- Writes a NEW CSV by default (does not overwrite input unless you pass --output_csv pointing to same path).

Input expectations
------------------
CSV must contain:
  - saved_filename  (preferred) OR filename
  - cx, cy, radius  (radius may be named 'r' in some older CSVs; this script can map it)

Images are expected to be 512x512 grayscale or color crops.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm


# -----------------------------
# Helpers
# -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def read_gray(image_path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img


def otsu_mask(img_gray: np.ndarray, blur: bool = True) -> np.ndarray:
    """Binary mask: pollen=1, background=0 (uint8 0/1)."""
    g = img_gray
    if blur:
        g = cv2.GaussianBlur(g, (3, 3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean small speckles / fill tiny holes
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    return (th > 0).astype(np.uint8)


def largest_contour(mask01: np.ndarray) -> Optional[np.ndarray]:
    cnts, _ = cv2.findContours((mask01 * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def mask_centroid(mask01: np.ndarray) -> Optional[Tuple[float, float]]:
    m = cv2.moments((mask01 * 255).astype(np.uint8))
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def shannon_entropy(values: np.ndarray, bins: int = 256) -> float:
    """Shannon entropy of grayscale intensities (base-2) for values array."""
    if values.size == 0:
        return np.nan
    hist = np.bincount(values.astype(np.uint8), minlength=bins).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def dist_transform(mask01: np.ndarray) -> np.ndarray:
    """Euclidean distance transform within mask (float)."""
    return ndimage.distance_transform_edt(mask01.astype(bool))


# -----------------------------
# Feature functions (10)
# -----------------------------

def feature_centroid_shift(mask01: np.ndarray, cx: float, cy: float, radius: float) -> float:
    """Normalized distance between mask centroid and provided (cx,cy)."""
    c = mask_centroid(mask01)
    if c is None:
        return np.nan
    mx, my = c
    d = math.hypot(mx - cx, my - cy)
    return float(d / max(radius, 1.0))


def feature_circularity(mask01: np.ndarray) -> float:
    """4πA / P^2 based on largest contour."""
    cnt = largest_contour(mask01)
    if cnt is None:
        return np.nan
    area = cv2.contourArea(cnt)
    per = cv2.arcLength(cnt, True)
    if per <= 0:
        return np.nan
    return float(4.0 * math.pi * area / (per * per))


def feature_convex_def_ratio(mask01: np.ndarray) -> float:
    """(HullArea - Area)/HullArea == 1 - solidity, based on largest contour."""
    cnt = largest_contour(mask01)
    if cnt is None:
        return np.nan
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return np.nan
    return float((hull_area - area) / hull_area)


def feature_edge_density(img_gray: np.ndarray, mask01: np.ndarray) -> float:
    """Edge pixels within mask / mask area."""
    edges = cv2.Canny(img_gray, 50, 150)
    edge_in = (edges > 0) & (mask01 > 0)
    area = int(mask01.sum())
    if area == 0:
        return np.nan
    return float(edge_in.sum() / area)


def feature_texture_entropy(img_gray: np.ndarray, mask01: np.ndarray) -> float:
    """Shannon entropy of grayscale intensities inside the pollen mask."""
    vals = img_gray[mask01 > 0]
    return shannon_entropy(vals)


def feature_radial_grad(img_gray: np.ndarray, cx: float, cy: float, radius: float) -> float:
    """
    Mean gradient magnitude in the annulus [0.5r, r] around (cx,cy),
    intersected with image bounds.
    """
    if radius <= 1:
        return np.nan
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    h, w = img_gray.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - cx, yy - cy)
    ring = (rr >= 0.5 * radius) & (rr <= radius)
    vals = mag[ring]
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals))


def feature_inscribed_ratio(mask01: np.ndarray, radius: float) -> float:
    """(max distance transform) / radius."""
    if radius <= 0:
        return np.nan
    dist = dist_transform(mask01)
    return float(dist.max() / max(radius, 1.0))


def feature_area_px(mask01: np.ndarray) -> int:
    return int(mask01.sum())


def feature_fractal_dim(mask01: np.ndarray) -> float:
    """Box-counting fractal dimension of the mask (binary)."""
    Z = (mask01 > 0).astype(np.uint8)
    if Z.sum() == 0:
        return np.nan

    # Use box sizes powers of 2
    p = min(Z.shape)
    n = int(np.floor(np.log2(p)))
    if n < 2:
        return np.nan
    sizes = 2 ** np.arange(n, 1, -1)

    def boxcount(img: np.ndarray, k: int) -> int:
        S = np.add.reduceat(np.add.reduceat(img, np.arange(0, img.shape[0], k), axis=0),
                            np.arange(0, img.shape[1], k), axis=1)
        return int(np.count_nonzero(S))

    counts = np.array([boxcount(Z, int(k)) for k in sizes], dtype=np.float64)
    # Avoid zeros
    mask = counts > 0
    if mask.sum() < 2:
        return np.nan
    coeffs = np.polyfit(np.log(sizes[mask]), np.log(counts[mask]), 1)
    return float(-coeffs[0])


def feature_spike_cnt(mask01: np.ndarray) -> float:
    """
    Approximate spike / indentation count using convexity defects on the largest contour.
    This is a pragmatic, reproducible proxy (depth-thresholded count).
    """
    cnt = largest_contour(mask01)
    if cnt is None or len(cnt) < 5:
        return np.nan

    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 5:
        return np.nan

    defects = cv2.convexityDefects(cnt, hull_idx)
    if defects is None:
        return 0.0

    # defects: [start, end, far, depth*256]
    depths = defects[:, 0, 3].astype(np.float32) / 256.0
    # threshold depth in pixels (tunable; keep modest)
    return float(np.sum(depths > 2.0))


# -----------------------------
# Z-score
# -----------------------------

def add_zscore(df: pd.DataFrame, col: str, suffix: str = "_zscore") -> None:
    vals = df[col].astype(float)
    mu = vals.mean(skipna=True)
    sd = vals.std(skipna=True, ddof=0)
    if sd == 0 or np.isnan(sd):
        df[col + suffix] = np.nan
    else:
        df[col + suffix] = (vals - mu) / sd


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True, help="Input CSV (from step 03).")
    ap.add_argument("--image_dir", required=True, help="Directory containing 512x512 crops.")
    ap.add_argument("--output_csv", default=None, help="Output CSV path. Default: <input>_features10.csv")
    ap.add_argument("--filter_crop_ok", action="store_true", help="Only process rows where crop_status=='ok' (if present).")
    ap.add_argument("--filter_center_ok", action="store_true", help="Only process rows where center_status=='ok' (if present).")
    ap.add_argument("--no_blur", action="store_true", help="Disable Gaussian blur before Otsu mask.")
    ap.add_argument("--debug_mask_dir", default=None, help="If set, saves mask images for rows processed (for sanity checks).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    img_dir = Path(args.image_dir)

    if args.output_csv:
        out_csv = Path(args.output_csv)
    else:
        out_csv = csv_path.with_name(csv_path.stem + "_features10.csv")

    df = pd.read_csv(csv_path)

    # Column normalization for radius
    if "radius" not in df.columns and "r" in df.columns:
        df["radius"] = df["r"]

    # Choose filename column
    fname_col = "saved_filename" if "saved_filename" in df.columns else ("filename" if "filename" in df.columns else None)
    if fname_col is None:
        raise ValueError("Input CSV must contain 'saved_filename' or 'filename' column.")

    required = ["cx", "cy", "radius"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Input CSV must contain '{c}' column (from step 03).")

    # Ensure output columns exist
    feature_cols = [
        "centroid_shift",
        "circularity",
        "convex_def_ratio",
        "edge_density",
        "texture_entropy",
        "radial_grad",
        "inscribed_ratio",
        "area_px",
        "fractal_dim",
        "spike_cnt",
        "feature_status",
    ]
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan if c != "feature_status" else ""

    blur = not args.no_blur

    if args.debug_mask_dir:
        debug_mask_dir = Path(args.debug_mask_dir)
        ensure_dir(debug_mask_dir)
    else:
        debug_mask_dir = None

    n_ok = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # optional filters
        if args.filter_crop_ok and "crop_status" in df.columns:
            if str(row.get("crop_status", "")).strip().lower() != "ok":
                df.at[idx, "feature_status"] = "skipped_not_ok_crop"
                continue

        if args.filter_center_ok and "center_status" in df.columns:
            if not str(row.get("center_status", "")).strip().lower().startswith("ok"):
                df.at[idx, "feature_status"] = "skipped_not_ok_center"
                continue

        fn = row.get(fname_col, "")
        if pd.isna(fn) or str(fn).strip() == "":
            df.at[idx, "feature_status"] = "missing_filename"
            continue

        image_path = img_dir / str(fn)
        if not image_path.exists():
            # Try if CSV already contains an absolute/relative path
            alt = Path(str(fn))
            if alt.exists():
                image_path = alt
            else:
                df.at[idx, "feature_status"] = "missing_image"
                continue

        img_gray = read_gray(image_path)
        if img_gray is None:
            df.at[idx, "feature_status"] = "read_failed"
            continue

        cx = safe_float(row.get("cx"))
        cy = safe_float(row.get("cy"))
        radius = safe_float(row.get("radius"))

        if np.isnan(cx) or np.isnan(cy) or np.isnan(radius):
            df.at[idx, "feature_status"] = "missing_cxcy_radius"
            continue

        # Mask
        mask01 = otsu_mask(img_gray, blur=blur)

        # Features
        try:
            df.at[idx, "centroid_shift"] = feature_centroid_shift(mask01, cx, cy, radius)
            df.at[idx, "circularity"] = feature_circularity(mask01)
            df.at[idx, "convex_def_ratio"] = feature_convex_def_ratio(mask01)
            df.at[idx, "edge_density"] = feature_edge_density(img_gray, mask01)
            df.at[idx, "texture_entropy"] = feature_texture_entropy(img_gray, mask01)
            df.at[idx, "radial_grad"] = feature_radial_grad(img_gray, cx, cy, radius)
            df.at[idx, "inscribed_ratio"] = feature_inscribed_ratio(mask01, radius)
            df.at[idx, "area_px"] = feature_area_px(mask01)
            df.at[idx, "fractal_dim"] = feature_fractal_dim(mask01)
            df.at[idx, "spike_cnt"] = feature_spike_cnt(mask01)

            df.at[idx, "feature_status"] = "ok"
            n_ok += 1

            if debug_mask_dir is not None:
                cv2.imwrite(str(debug_mask_dir / f"mask_{idx:06d}.png"), (mask01 * 255).astype(np.uint8))

        except Exception as e:
            df.at[idx, "feature_status"] = f"error:{type(e).__name__}"

    # Z-score columns
    std_cols = [
        "radius",
        "centroid_shift",
        "circularity",
        "convex_def_ratio",
        "edge_density",
        "texture_entropy",
        "radial_grad",
        "inscribed_ratio",
        "area_px",
        "fractal_dim",
        "spike_cnt",
    ]
    for col in std_cols:
        if col in df.columns:
            add_zscore(df, col)

    df.to_csv(out_csv, index=False)
    print(f"\n✅ Feature table saved -> {out_csv}")
    print(f"Rows with feature_status=='ok': {n_ok}/{len(df)}")


if __name__ == "__main__":
    main()
