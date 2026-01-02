# SVM_learn_final0907.py  改訂版
# ------------------------------------------------------------
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_predict
)
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance
from joblib import dump

# ▼--- ファイルパスとクラス設定 ---------------------------------
CSV_PATH = r"pollen_10features_from_csv_reverse_zscore.csv"
IMG_CLASSES = ["cedar", "cypre", "dust", "cedar_brst", "cypre_brst"]
# ----------------------------------------------------------------

# Candidates to use (here, 9 features are used in the final model)
feature_cols_9 = [
    "radius_zscore",
    "spike_cnt_zscore",
    "centroid_shift_zscore",
    "circularity_zscore",
    "convex_def_ratio_zscore",
    "texture_entropy_zscore",
    "radial_grad_zscore",
    "inscribed_ratio_zscore",
    "area_px_zscore",
]

feature_cols_8 = [
    "radius_zscore",
    "spike_cnt_zscore",
    "centroid_shift_zscore",
    "circularity_zscore",
    "texture_entropy_zscore",
    "radial_grad_zscore",
    "inscribed_ratio_zscore",
    "area_px_zscore",
]

def data_retrieve(csv_path, classes, feature_names):
    """Extract and return only the target class and feature columns from CSV"""
    df = pd.read_csv(csv_path)
    df = df[df["type"].isin(classes)].reset_index(drop=True)

    # Numeric labeling (depending on IMG_CLASSES list)
    label_map = {name: i for i, name in enumerate(classes)}
    df["label"] = df["type"].map(label_map)

    X = df[feature_names].copy()
    y = df["label"].copy()

    # If there is a defect, sync drop
    mask = ~X.isna().any(axis=1)
    X = X.loc[mask]
    y = y.loc[mask]

    return X, y

def machine_learning_pollen(features: pd.DataFrame, labels: pd.Series, suffix="1231"):
    """
    Output confusion matrix and report using 5-fold Stratified CV.
    Here, we use a "fixed hyperparameter" SVM for the purpose of creating the graph.
    """
    k = features.shape[1]
    fig_name = f"confusion_matrix_svm_5foldCV_{k}features_{suffix}.png"

    # Model: Standardization + RBF-SVM (including imbalance measures)
    model = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, features, labels, cv=cv)

    print(f"\n📋 Classification Report ({k} features, 5-fold CV):\n")
    print(classification_report(labels, y_pred, target_names=IMG_CLASSES, digits=3))

    cm = confusion_matrix(labels, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=IMG_CLASSES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    plt.title(f"Confusion Matrix (SVM, 5-fold CV, {k} features)")
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)
    plt.close()
    print(f"✅  Confusion matrix saved → {fig_name}")

def fit_final_estimator(X: pd.DataFrame, y: pd.Series):

    # Using all data X and y, optimize C/gammma/class_weight in the innne 3-fold and 
    # return a refitted learner with the best parameters(RBF-SVM, probability = True)

    pipe = SVC(kernel="rbf", probability=True, cache_size=2000)
    
    param_grid = {
        "C": [1, 3, 10, 30],
        "gamma": ["scale", 0.1, 0.03],
        "class_weight": [None, "balanced"],
    }
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    gs = GridSearchCV(
        pipe, param_grid, cv=inner,
        scoring="f1_macro", n_jobs=-1, refit=True, verbose=0
    )
    gs.fit(X, y)
    print("[final] best params:", gs.best_params_)
    return gs.best_estimator_

def plot_permutation_importance(
    estimator, X: pd.DataFrame, y: pd.Series, feature_names,
    out_png="perm_importance_bar_1231.png",
    out_csv="perm_importance_table_1231.csv",
    scoring="f1_macro", n_repeats=50, random_state=42,
    n_jobs=-1, dpi=300
):
    """
    Calculate permutation_importance and save a PNG of horizontal bars and error bars.
    Evaluation data can be the same as training data (recommended to note "for interpretation purposes").
    """
    print("[PI] computing permutation importance...")
    result = permutation_importance(
        estimator, X, y,
        scoring=scoring, n_repeats=n_repeats,
        random_state=random_state, n_jobs=n_jobs
    )
    
    mean_all = result.importances_mean
    std_all  = result.importances_std
    p05_all  = np.percentile(result.importances, 5, axis=1)
    p95_all  = np.percentile(result.importances, 95, axis=1)

    # In descending order of mean (in descending order of importance)
    order_desc = np.argsort(mean_all)[::-1]

    names = np.array(feature_names)[order_desc]
    mean = mean_all[order_desc]
    std  = std_all[order_desc]
    p05  = p05_all[order_desc]
    p95  = p95_all[order_desc]

    names_pretty = [n[:-7] if n.endswith("_zscore") else n for n in names]

    # Feature and mean/std/p05/p95 are saved in the same order
    df_pi = pd.DataFrame({
        "rank": np.arange(1, len(names) + 1),
        "feature": names,
        "feature_pretty": names_pretty,
        "mean": mean,
        "std": std,
        "p05": p05,
        "p95": p95,
    })
    df_pi.to_csv(out_csv, index=False)
    print(f"✅  Permutation importance table → {out_csv}")

    # Plot
    plt.figure(figsize=(7, 5), dpi=dpi)
    ypos = np.arange(len(names_pretty))[::-1]  
    plt.barh(ypos, mean[::-1], height=0.6)
    plt.errorbar(mean[::-1], ypos, xerr=std[::-1], fmt="none", capsize=3, linewidth=1)
    plt.axvline(0, ls="--", lw=1)
    plt.yticks(ypos, names_pretty[::-1])
    plt.xlabel("Δ macro-F1 (mean ± SD over permutations)")
    plt.title("Permutation importance")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()
    print(f"✅  PI figure saved → {out_png}")

def save_estimator(estimator, feature_names, suffix = "1231", out_path=None):
    if out_path is None:
        out_path = f"svm_final_{len(feature_names)}features_{suffix}.joblib"
    dump(estimator, out_path)
    print(f"✅  Model saved → {out_path}")
    return out_path

def main():
    start = time.time()
    suffix = "1231"

    X, y = data_retrieve(CSV_PATH, IMG_CLASSES, feature_cols_9)
 
    # confusion matrix (make a figure)
    machine_learning_pollen(X, y, suffix=suffix)

    # best model（Optimize with inner CV and refit with all data）
    final_estimator = fit_final_estimator(X, y)

    # PI
    plot_permutation_importance(
        final_estimator, X, y, feature_cols_9,
        out_png=f"perm_importance_bar_{suffix}.png",
        out_csv=f"perm_importance_table_{suffix}.csv",
        scoring="f1_macro", n_repeats=50, random_state=42, n_jobs=-1
    )

    save_estimator(final_estimator, feature_cols_9, suffix=suffix)

    print(f"\nElapsed: {time.time() - start:.1f} sec")

if __name__ == "__main__":
    main()
