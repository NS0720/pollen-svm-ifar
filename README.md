# IFAR Pollen Image Pipeline (01–07)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18129958.svg)](https://doi.org/10.5281/zenodo.18129958)

This repository contains the full, reproducible analysis pipeline used in the IFAR manuscript:

**raw 400× microscope images → 512×512 crops → engineered features (with z-scores) → SVM evaluation → permutation importance**

> **Note (manual step):** Class labels (`type`) are assigned by expert visual inspection of the cropped images.  
> This step is intentionally manual and is not automated in this repository.

---

## Requirements

- Python **3.10+** (3.11 also OK)
- Packages: see `requirements.txt`

Install:

```bash
pip install -r requirements.txt
```

---

## Input data

### Raw images
Put your original microscope images (e.g., `.jpg`) into a folder, for example:

```
pollen_picture_folder/
  IMG_0001.jpg
  IMG_0002.jpg
  ...
```

### Class labels (manual)
The downstream ML scripts (05–07) expect a CSV column:

- `type` ∈ `{cedar, cypre, dust, cedar_brst, cypre_brst}`

You will add/fill this column manually at the indicated step below.

---

## Quick workflow (no batch files)

Create output folders once:

```cmd
mkdir out01 out02 out03 out04
```

### Step 01 — Detect circles (Hough + DBSCAN) and optionally crop rough patches
Script: `01_round_detect_400_dbscan_IFAR_v2.py`

Example:

```cmd
python 01_round_detect_400_dbscan_IFAR_v2.py ^
  --input_dir pollen_picture_folder ^
  --output_dir out01 ^
  --write_csv ^
  --csv_name circle_data.csv ^
  --deterministic_names
```

**Main outputs (in `out01/`):**
- Rough crops (variable size, for quick sanity check)
- `circle_data.csv` (at least: `original_filename, x, y, r, cropped_filename`)

---

### Step 02 — Crop **exact 512×512** tiles from original images
Script: `02_crop512centered_IFAR_v2.py`

Example:

```cmd
python 02_crop512centered_IFAR_v2.py ^
  --csv_path out01\circle_data.csv ^
  --image_dir pollen_picture_folder ^
  --output_dir out02 ^
  --size 512 ^
  --out_csv out02\crop512_log.csv ^
  --name_mode original_index
```

**Main outputs (in `out02/`):**
- 512×512 crops (padded to fixed size when near borders)
- `crop512_log.csv` (adds: `saved_filename`, `crop_status`)

`crop_status` helps you filter valid rows:
- `ok` (crop saved)
- `missing_image`, `read_failed`, `write_failed`, ...

---

### Manual step — Fill the `type` column by visual inspection
Open `out02\crop512_log.csv` in Excel and **fill/verify** the `type` column for rows with `crop_status == "ok"`.

Save as a new file (recommended):

```
out02\crop512_log_labeled.csv
```

---

### Step 03 — Estimate pollen center `(cx, cy)` from the 512×512 crop
Script: `03_estimate_center_cxcy_IFAR_v3.py`

Example:

```cmd
python 03_estimate_center_cxcy_IFAR_v3.py ^
  --csv_path out02\crop512_log_labeled.csv ^
  --image_dir out02 ^
  --filter_crop_ok ^
  --output_csv out03\center.csv
```

**Main outputs:**
- `out03\center.csv` (adds: `cx, cy, gx, gy, dist, offset_ratio, center_status`)

`center_status` indicates success/failure per row (`ok`, `no_contour`, `read_failed`, ...).

Optional debug overlays:

```cmd
python 03_estimate_center_cxcy_IFAR_v3.py ^
  --csv_path out02\crop512_log_labeled.csv ^
  --image_dir out02 ^
  --filter_crop_ok ^
  --debug_overlay_dir out03\debug_overlay ^
  --output_csv out03\center.csv
```

---

### Step 04 — Generate 10 engineered features + `*_zscore`
Script: `04_make_features10_IFAR_v3.py`

Example:

```cmd
python 04_make_features10_IFAR_v3.py ^
  --csv_path out03\center.csv ^
  --image_dir out02 ^
  --filter_crop_ok ^
  --filter_center_ok ^
  --output_csv out04\features10_zscore.csv
```

**Main outputs:**
- `out04\features10_zscore.csv` (adds feature columns + `*_zscore` columns + `feature_status`)

This CSV is the **final dataset** used by scripts 05–07.  
If you want to use the same filename as the manuscript scripts, either:
- rename it to `pollen_10features_from_csv_reverse_zscore.csv`, or
- edit `CSV_PATH` at the top of 05/06/07.

---

## ML / validation (scripts 05–07)

These scripts read the final feature CSV (default: `pollen_10features_from_csv_reverse_zscore.csv`).  
They do not currently use command-line arguments; configure by editing variables at the top of each script.

### Step 05 — Permutation importance + final SVM fit (interpretability)
Script: `05_Permutation_Importance_IFAR.py`

Outputs (examples):
- `perm_importance_bar_{suffix}.png`
- `perm_importance_table_{suffix}.csv`
- `svm_final_{...}_{suffix}.joblib`
- `confusion_matrix_svm_5foldCV_{k}features_{suffix}.png`

> **Important:** permutation importance here is for **interpretation**, not for unbiased performance estimation.

---

### Step 06 — “Drop-column” style feature-set comparison (11→9)
Script: `06_drop_column_IFAR.py`

Prints a table of mean±SD outer-CV scores for several feature sets.

---

### Step 07 — Nested CV (5×5) evaluation + OOF ROC/PR/CM
Script: `07_nestedCV_IFAR.py`

Outputs include:
- `confusion_matrix_oof_9z_*.png`
- `roc_oof_9z_*.png`
- `pr_oof_9z_with_AUPRC_baseline_*.png`

These figures correspond to the **primary** (unbiased) performance evaluation.

---

## Example images (included)

Ten 400× magnification pollen images are included in `pollen_picture_folder/` as a minimal dataset for reproduction.

---

## Notes on reproducibility (DOI-friendly)

- Steps 02–04 write results **per row** and add `*_status` columns to avoid silent misalignment.
- Keep each step output in a separate folder (`out01/`…`out04/`) to prevent accidental overwrites.
- Manual labeling is expected; save the labeled CSV under a new name (e.g., `crop512_log_labeled.csv`).

---

## Authorship / license

- Author: **Nobuyoshi Suzuki**
- License: **MIT License** (see `LICENSE`)

---

## Citation

If you use this code, please cite the Zenodo record.

- **Version DOI (v1.0.1):** 10.5281/zenodo.18130679  
- **Concept DOI (all versions):** 10.5281/zenodo.18129958  

Suggested citation (APA):

Suzuki, N. (2026). *NS0720/pollen-svm-ifar: v1.0.1 - IFAR pipeline (pollen detection → features → SVM) (v1.0.2).* Zenodo. https://doi.org/10.5281/zenodo.18130679
