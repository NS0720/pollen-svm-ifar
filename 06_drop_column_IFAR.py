from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.svm import SVC
import numpy as np, pandas as pd

CSV_PATH = r"pollen_10features_from_csv_reverse_zscore.csv"
IMG_CLASSES = ["cedar", "cypre", "dust", "cedar_brst", "cypre_brst"]


# Load data and extract only the necessary rows
df = pd.read_csv(CSV_PATH)
df = df[df["type"].isin(IMG_CLASSES)].reset_index(drop=True)

feat_cols = [c for c in df.columns if c.endswith("_zscore")]

X = df[feat_cols]  # all Z-score row
y = df['type']
core9_cols = ["radius_zscore","centroid_shift_zscore","circularity_zscore","spike_cnt_zscore","texture_entropy_zscore","area_px_zscore","radial_grad_zscore","inscribed_ratio_zscore","convex_def_ratio_zscore"]
A_11_cols = core9_cols + ["edge_density_zscore", "fractal_dim_zscore"]
sets = {
  "A_11": A_11_cols,
  "B_no_fractal": [c for c in A_11_cols if c != "fractal_dim_zscore" ],
  "C_no_spike":   [c for c in A_11_cols if c !="spike_cnt_zscore" ],
  "D_no_edge":    [c for c in A_11_cols if c !="edge_density_zscore" ],
  "E_core9":      core9_cols   
}

outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
svc = SVC(kernel="rbf", probability=False, class_weight="balanced")
param = {"C":[0.3,1,3,10,30], "gamma":[0.003,0.01,0.03,"scale"]}

rows=[]
for name, cols in sets.items():
    gs = GridSearchCV(
        estimator=svc,
        param_grid=param,
        cv=inner,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
        verbose=0
    )
    # outer CV score 
    scores = cross_val_score(gs, X[cols], y, cv=outer, scoring="f1_macro", n_jobs=-1)
    rows.append({
        "set": name,
        "k": len(cols),
        "f1_macro_mean": float(scores.mean()),
        "f1_macro_sd": float(scores.std(ddof=1)),
    })

tab = pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)
print(tab)
