#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E_core9 (all *_zscore) + nested 5x5 CV + OOF confusion matrix / ROC / PR
Classes used: cedar, cypre, dust, cedar_brst, cypre_brst
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from matplotlib.ticker import StrMethodFormatter

# ===== 0) Input Settings =====
CSV_PATH = r"pollen_10features_from_csv_reverse_zscore.csv"  
USE_CLASSES = ["cedar", "cypre", "dust", "cedar_brst", "cypre_brst"]

# 9 features (all *_zscore)
FEATURES_9_Z = [
    "radius_zscore",   
    "spike_cnt_zscore",
    "centroid_shift_zscore",
    "circularity_zscore",
    "convex_def_ratio_zscore",
    "texture_entropy_zscore",
    "radial_grad_zscore",
    "inscribed_ratio_zscore",
    "area_px_zscore"
]

def mean_sd_ci(scores, alpha=0.05):
    """
    scores: list/np.array of outer-fold scores (n=5 etc.)
    returns: mean, sd, (ci_low, ci_high), df
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    mean = scores.mean()
    sd = scores.std(ddof=1)
    df = n - 1
    try:
        from scipy.stats import t
        tcrit = t.ppf(1 - alpha/2, df)
    except Exception:
        # fallback: df=4のときの近似（n=5想定）
        tcrit = 2.776 if df == 4 else 1.96
    half = tcrit * sd / np.sqrt(n)
    return mean, sd, (mean - half, mean + half), df

def paired_pvalue(a, b):
    """
    Paired test across outer folds.
    a,b: arrays of length n (same folds)
    returns: p-value (two-sided)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    try:
        from scipy.stats import ttest_rel
        return float(ttest_rel(a, b).pvalue)
    except Exception:
        return np.nan

# ===== 1) Data Loading & Filtering=====
df = pd.read_csv(CSV_PATH)

# Limited to only 5 classes used
df = df[df["type"].isin(USE_CLASSES)].reset_index(drop=True)

# Required column check
missing = [c for c in FEATURES_9_Z if c not in df.columns]
if missing:
    raise ValueError(f"There is a missing column in the CSV: {missing}")

X_df = df[FEATURES_9_Z].copy()
y_df = df["type"].copy()

# Missing data exclusion (synchronize X and y)
mask = ~X_df.isna().any(axis=1)
X = X_df.loc[mask].to_numpy(dtype=float)
y = y_df.loc[mask].to_numpy()

classes = np.array(USE_CLASSES, dtype=object)  # Fix class order

print("Counts after mask:")
print(pd.Series(y).value_counts()[list(classes)])

# ===== 2) nested CV settings =====
K_OUT, K_IN = 5, 5
outer = StratifiedKFold(n_splits=K_OUT, shuffle=True, random_state=42)

pipe = SVC(kernel="rbf", class_weight="balanced", probability=False)

param_grid = {
    "C":     [0.3, 1, 3, 10, 30],
    "gamma": [0.003, 0.01, 0.03, "scale"],
}

# OOF container (backfill all outer test predictions)
y_pred_oof  = np.empty_like(y, dtype=object)
y_score_oof = np.zeros((len(y), len(classes)), dtype=float)
outer_scores = []

for tr_idx, te_idx in outer.split(X, y):
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    inner = StratifiedKFold(n_splits=K_IN, shuffle=True, random_state=1)
    gs = GridSearchCV(
        pipe, param_grid, cv=inner,
        scoring="f1_macro", n_jobs=-1, refit=True, verbose=0
    )
    gs.fit(X_tr, y_tr)
    best = gs.best_estimator_

    # Out-of-fold (OOF) predeictions
    y_pred = best.predict(X_te)
    y_pred_oof[te_idx] = y_pred

    # Multi-class scores: decision_function (OvR)
    scores = best.decision_function(X_te)  # shape: (n_te, n_classes)
    if scores.ndim == 1:  # Two-value insurance
        scores = np.vstack([-scores, scores]).T

    # Match the class order in the learner to our classes (just to be safe)
    if hasattr(best, "classes_"):
        order = [np.where(best.classes_ == c)[0][0] for c in classes]
        scores = scores[:, order]

    y_score_oof[te_idx, :] = scores

    # outer test macro-F1
    outer_scores.append(f1_score(y_te, y_pred, average="macro"))

mean, sd, (ci_low, ci_high), df = mean_sd_ci(outer_scores)
print(f"macro-F1 (outer OOF): {mean:.3f} (SD={sd:.3f}; 95% CI {ci_low:.3f}–{ci_high:.3f}), df={df}")

print(f"[Nested CV] macro-F1 (outer test): {np.mean(outer_scores):.3f} +/- {np.std(outer_scores):.3f}")

# ===== 3) confusion matrix (OOF) =====
cm = confusion_matrix(y, y_pred_oof, labels=classes)
print("\nClassification report (OOF):")
print(classification_report(y, y_pred_oof, labels= classes, target_names=classes, digits=3, zero_division=0))

def plot_cm(cm, classes, normalize=False, fname="confusion_matrix_oof_9z_1231.png"):
    """Plot cm with annotations. Normalize=True displays percentages in the row direction."""
    plt.figure(figsize=(7.2, 6.2))
    if normalize:
        cm_to_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cmap_label = "Proportion"
        fmt = ".2f"
    else:

        
        cm_to_plot = cm
        cmap_label = "Count"
        fmt = "d"

    im = plt.imshow(cm_to_plot, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = plt.colorbar(im)
    cbar.set_label(cmap_label)

    plt.title("Confusion Matrix (nested-CV OOF, 9 z-score features)")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Switch text color at threshold
    thresh = cm_to_plot.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_to_plot[i, j]
            txt = f"{val:{fmt}}"
            plt.text(j, i, txt,
                     ha="center", va="center",
                     color="white" if val > thresh else "black", fontsize=11)

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()

# If you want to save both the number version and the percentage version, call it twice.
plot_cm(cm, classes, normalize=False, fname="confusion_matrix_oof_9z_counts_1231.png")
plot_cm(cm, classes, normalize=True,  fname="confusion_matrix_oof_9z_ratio_1231.png")

# ===== 4) ROC / PR (OOF) =====
Ybin = label_binarize(y, classes=classes)

# ROC
plt.figure(figsize=(6.2, 6.2))
for k, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(Ybin[:, k], y_score_oof[:, k])
    auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{cls} (AUC={auc_val:.2f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC (nested-CV OOF, 9 z-score features)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("roc_oof_9z_1231.png", dpi=300)
plt.close()

# PR
plt.figure(figsize=(6.5, 6.2))

aps = []
baselines = []
for k, cls in enumerate(classes):
    prec, rec, _ = precision_recall_curve(Ybin[:, k], y_score_oof[:, k])
    ap =  average_precision_score(Ybin[:, k], y_score_oof[:, k])
    baseline = Ybin[:, k].mean()
    plt.plot(rec, prec, label=f"{cls} (AUPRC={ap:.2f},baseline={baseline:.3f})")
    aps.append(ap)
    baselines.append(baseline)

ap_micro = average_precision_score(Ybin, y_score_oof, average= "micro")
ap_macro = average_precision_score(Ybin, y_score_oof, average= "macro")

plt.legend(frameon=False, loc="lower left")

plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall curves")
plt.tight_layout()
plt.savefig("pr_oof_9z_with_AUPRC_baseline_1231.png", dpi=300)
plt.close()