#!/usr/bin/env python3
"""
run_ml_pipeline.py

Run with no args:
  python run_ml_pipeline.py

What it does:
1) Reference reproduction run on dataset.csv (paper-provided full descriptor matrix; p=357, n=52).
2) Your reproduction run on my_dataset.csv (your computed descriptors), merged with ddGglyc from dataset.csv.
3) Implements the Supplementary ML workflow:
   - Center + scale descriptors (StandardScaler)
   - PCA-ranking: rank PCs by |corr(PC, ddGglyc)|
   - Select descriptors "in the space of top PCs" with an arbitrary cut to keep feature count < n
   - LASSO via LassoLarsIC to select minimal feature subset
   - Linear model evaluated via Leave-One-Out CV (LOO)
   - Random Forest evaluated via OOB score; simple tree-count search

Outputs:
  results_ml/reference/*
  results_ml/user/*
  results_ml/diagnostics/*

Dependencies:
  pandas, numpy, scikit-learn, matplotlib
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_CSV = BASE_DIR / "dataset.csv"        # paper reference CSV
MY_DATASET_CSV = BASE_DIR / "my_dataset.csv"  # your computed CSV
OUT_DIR = BASE_DIR / "results_ml"


# ----------------------------
# Helpers
# ----------------------------
META_COLS = ["Unnamed: 0", "name", "interactor", "ddGglyc", "ddGglyc error"]

def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all non-id columns to numeric where possible."""
    out = df.copy()
    for c in out.columns:
        if c in ("name", "interactor"):
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _feature_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in META_COLS]
    # keep only numeric-ish columns (after conversion, dtype will be float/int)
    cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    return cols

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _rmse(y_true, y_pred) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))

def _save_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def _plot_parity(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure()
    plt.scatter(y_true, y_pred)
    # y=x
    mn = float(np.nanmin([y_true.min(), y_pred.min()]))
    mx = float(np.nanmax([y_true.max(), y_pred.max()]))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True ddGglyc")
    plt.ylabel("Predicted ddGglyc")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def _plot_bar(values: np.ndarray, labels: List[str], title: str, out_path: Path, top_k: int = 25) -> None:
    # take top_k by absolute value (useful for correlations/importances)
    idx = np.argsort(np.abs(values))[::-1][:top_k]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(idx)), values[idx])
    plt.xticks(range(len(idx)), [labels[i] for i in idx], rotation=75, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------
# Feature selection = paper-style (robust)
# ----------------------------
@dataclass
class SelectionReport:
    n_samples: int
    n_features_in: int
    top_pcs: List[int]
    top_pc_corrs: List[float]
    max_features_cap: int
    selected_by_pc_loading: List[str]
    selected_by_lasso: List[str]
    lasso_alpha: float

def pca_ranking_and_select(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    max_features_cap: int,
    random_state: int = 42,
) -> Tuple[List[str], SelectionReport]:
    """
    Implements the Supplementary logic:

    - Center+scale descriptors
    - PCA
    - Rank PCs by |corr(PC_score, y)|
    - Consider the top 3 PCs
    - Select descriptors "in the space of top PCs" via large loadings in those PCs
      (we use sum(loadings^2) across top PCs)
    - Keep at most max_features_cap (< n to avoid underdetermined stats)
    - Run LassoLarsIC on that reduced set to get a minimal subset
    """
    n = X.shape[0]
    p = X.shape[1]

    # preprocess
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(imp.fit_transform(X))

    # PCA
    n_comp = min(n - 1, p)
    pca = PCA(n_components=n_comp, random_state=random_state)
    scores = pca.fit_transform(Xs)

    # PC ranking by correlation to y
    pc_corrs = []
    for k in range(scores.shape[1]):
        c = np.corrcoef(scores[:, k], y)[0, 1]
        if np.isnan(c):
            c = 0.0
        pc_corrs.append(float(c))
    pc_corrs = np.array(pc_corrs)
    top_pcs = np.argsort(np.abs(pc_corrs))[::-1][: min(3, len(pc_corrs))]

    # descriptor "importance" in the space of top PCs (squared loadings sum)
    loadings = pca.components_[top_pcs, :]  # shape: (<=3, p)
    feat_score = np.sum(loadings ** 2, axis=0)  # shape: (p,)

    order = np.argsort(feat_score)[::-1]
    reduced = [feature_names[i] for i in order[:max_features_cap]]

    # LASSO on reduced
    X_red = X[:, [feature_names.index(c) for c in reduced]]
    X_red_s = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X_red))

    # IMPORTANT: LassoLarsIC needs n_samples > n_features + 1 (intercept)
    # So we enforce cap <= n-2 earlier.
    lasso = LassoLarsIC(criterion="aic")
    lasso.fit(X_red_s, y)

    coefs = lasso.coef_
    selected = [f for f, c in zip(reduced, coefs) if abs(c) > 1e-12]

    # Fallback if IC shrinks everything to 0 (can happen on some inputs)
    if len(selected) == 0:
        # take top-3 correlated features (simple & stable)
        corr = []
        for c in reduced:
            xv = X[:, feature_names.index(c)]
            cc = np.corrcoef(xv, y)[0, 1]
            if np.isnan(cc):
                cc = 0.0
            corr.append(cc)
        corr = np.array(corr)
        top3 = np.argsort(np.abs(corr))[::-1][: min(3, len(corr))]
        selected = [reduced[i] for i in top3]

    report = SelectionReport(
        n_samples=int(n),
        n_features_in=int(p),
        top_pcs=[int(x) for x in top_pcs],
        top_pc_corrs=[float(pc_corrs[i]) for i in top_pcs],
        max_features_cap=int(max_features_cap),
        selected_by_pc_loading=reduced,
        selected_by_lasso=selected,
        lasso_alpha=float(lasso.alpha_),
    )
    return selected, report


# ----------------------------
# Models
# ----------------------------
def eval_linear_loocv(X: np.ndarray, y: np.ndarray) -> Tuple[Dict, np.ndarray]:
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LinearRegression()),
    ])
    loo = LeaveOneOut()
    y_pred = cross_val_predict(pipe, X, y, cv=loo)
    metrics = {
        "rmse": _rmse(y, y_pred),
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
        "cv": "LOO",
    }
    return metrics, y_pred

def fit_rf_oob_search(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators_grid: Tuple[int, ...] = (200, 500, 1000, 2000),
    max_depth_grid: Tuple[Optional[int], ...] = (None, 5, 10),
    max_features_grid: Tuple[object, ...] = ("sqrt", 0.5, 1.0),
) -> Tuple[Dict, np.ndarray]:
    # Scale to mirror “centered and scaled” step (paper says scaler used before modeling)
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(imp.fit_transform(X))

    best = None
    best_oob = -1e9

    for n_est in n_estimators_grid:
        for md in max_depth_grid:
            for mf in max_features_grid:
                rf = RandomForestRegressor(
                    n_estimators=int(n_est),
                    random_state=42,
                    bootstrap=True,
                    oob_score=True,
                    n_jobs=-1,
                    max_depth=md,
                    max_features=mf,
                    min_samples_leaf=1,
                )
                rf.fit(Xs, y)
                oob = float(rf.oob_score_)
                if oob > best_oob:
                    best_oob = oob
                    best = (rf, {"n_estimators": n_est, "max_depth": md, "max_features": mf}, imp, scaler)

    rf, params, imp, scaler = best
    oob_pred = getattr(rf, "oob_prediction_", None)
    if oob_pred is None:
        oob_pred = rf.predict(scaler.transform(imp.transform(X)))

    mask = ~np.isnan(oob_pred)
    metrics = {
        "oob_r2": float(best_oob),
        "rmse": _rmse(y[mask], oob_pred[mask]),
        "mae": float(mean_absolute_error(y[mask], oob_pred[mask])),
        "r2": float(r2_score(y[mask], oob_pred[mask])),
        "cv": "OOB",
        "best_params": params,
    }
    return metrics, oob_pred


# ----------------------------
# Diagnostics: compare your descriptors vs reference
# ----------------------------
def descriptor_diff_report(ref: pd.DataFrame, usr: pd.DataFrame, out_dir: Path) -> None:
    """
    Compares overlapping descriptor columns between:
      - ref (dataset.csv values)
      - usr (my_dataset.csv values merged to ddGglyc)
    """
    _safe_mkdir(out_dir)

    common = sorted(set(_feature_cols(ref)).intersection(set(_feature_cols(usr))))
    if not common:
        (out_dir / "diff_summary.txt").write_text("No overlapping descriptor columns found.", encoding="utf-8")
        return

    r = _to_numeric_df(ref[["name", "interactor"] + common]).set_index(["name", "interactor"])
    u = _to_numeric_df(usr[["name", "interactor"] + common]).set_index(["name", "interactor"])

    joined = r.join(u, lsuffix="_ref", rsuffix="_usr", how="inner")

    rows = []
    for c in common:
        a = joined[f"{c}_ref"].values.astype(float)
        b = joined[f"{c}_usr"].values.astype(float)
        mask = ~np.isnan(a) & ~np.isnan(b)
        if mask.sum() == 0:
            rows.append({"feature": c, "n": 0, "mae": np.nan, "rmse": np.nan, "max_abs": np.nan})
            continue
        diff = b[mask] - a[mask]
        rows.append({
            "feature": c,
            "n": int(mask.sum()),
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(math.sqrt(np.mean(diff ** 2))),
            "max_abs": float(np.max(np.abs(diff))),
        })

    rep = pd.DataFrame(rows).sort_values(["rmse", "mae"], ascending=False)
    rep.to_csv(out_dir / "descriptor_diff_per_feature.csv", index=False)

    # also a quick row-wise aggregate diff
    row_mae = []
    for idx in joined.index:
        diffs = []
        for c in common:
            ar = joined.loc[idx, f"{c}_ref"]
            au = joined.loc[idx, f"{c}_usr"]
            if pd.notna(ar) and pd.notna(au):
                diffs.append(float(abs(au - ar)))
        row_mae.append({"name": idx[0], "interactor": idx[1], "mean_abs_diff": float(np.mean(diffs)) if diffs else np.nan})

    pd.DataFrame(row_mae).to_csv(out_dir / "descriptor_diff_per_system.csv", index=False)


# ----------------------------
# Main runner
# ----------------------------
def run_one(
    df: pd.DataFrame,
    label: str,
    out_dir: Path,
    selection_cap: Optional[int] = None,
) -> None:
    _safe_mkdir(out_dir)

    df = _strip_cols(df)
    df = _to_numeric_df(df)

    if "ddGglyc" not in df.columns:
        raise ValueError(f"{label}: ddGglyc not found in dataframe")

    y = df["ddGglyc"].values.astype(float)
    feat_cols = _feature_cols(df)

    if len(feat_cols) == 0:
        raise ValueError(f"{label}: no numeric descriptor columns detected")

    X = df[feat_cols].values.astype(float)
    n = X.shape[0]

    # sanity: ensure there is enough non-missing data
    nonmissing_rows = np.sum(~np.isnan(X).all(axis=1))
    if nonmissing_rows < max(10, int(0.5 * n)):
        print(f"[{label}] WARNING: too many missing descriptor rows ({nonmissing_rows}/{n} have any values).")
        print(f"[{label}] Run your descriptor scripts first (Part A/B/C) to fill my_dataset.csv.")
        # still save a small note
        (out_dir / "WARNING.txt").write_text(
            f"Too many missing descriptor rows: {nonmissing_rows}/{n} have any values.\n"
            f"Run descriptor computation scripts first.\n",
            encoding="utf-8"
        )
        return

    # selection cap: paper says reduce p below n (arbitrary limit)
    # We enforce cap <= n-2 (required for LassoLarsIC with intercept).
    if selection_cap is None:
        selection_cap = max(5, n - 2)

    selected_feats, sel_report = pca_ranking_and_select(
        X=X, y=y, feature_names=feat_cols, max_features_cap=selection_cap
    )

    # save selection report
    _save_json(out_dir / "selection_report.json", asdict(sel_report))
    (out_dir / "selected_features.txt").write_text("\n".join(selected_feats) + "\n", encoding="utf-8")

    X_sel = df[selected_feats].values.astype(float)

    # Linear: LOO-CV
    lin_metrics, y_pred_lin = eval_linear_loocv(X_sel, y)
    _save_json(out_dir / "linear_loocv_metrics.json", lin_metrics)
    _plot_parity(y, y_pred_lin, f"{label} — Linear (LOO-CV)", out_dir / "linear_parity.png")

    # RF: OOB
    rf_metrics, y_pred_oob = fit_rf_oob_search(X_sel, y)
    _save_json(out_dir / "rf_oob_metrics.json", rf_metrics)
    _plot_parity(y, y_pred_oob, f"{label} — RandomForest (OOB)", out_dir / "rf_oob_parity.png")

    # Save predictions
    pred_df = df[["name", "interactor", "ddGglyc"]].copy()
    pred_df["pred_linear_loocv"] = y_pred_lin
    pred_df["pred_rf_oob"] = y_pred_oob
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    # Summary print
    print(f"\n[{label}] Features in: {len(feat_cols)}")
    print(f"[{label}] Selected features (LASSO): {len(selected_feats)}")
    print(f"[{label}] Linear LOO:  R2={lin_metrics['r2']:.3f}  RMSE={lin_metrics['rmse']:.3f}")
    print(f"[{label}] RF OOB:     R2={rf_metrics['r2']:.3f}  RMSE={rf_metrics['rmse']:.3f}  best={rf_metrics['best_params']}")

def main():
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Missing: {DATASET_CSV}")
    if not MY_DATASET_CSV.exists():
        raise FileNotFoundError(f"Missing: {MY_DATASET_CSV}")

    _safe_mkdir(OUT_DIR)

    # --- Reference run (paper dataset)
    ref = _strip_cols(pd.read_csv(DATASET_CSV))
    ref_out = OUT_DIR / "reference"
    print("== Running reference ML on dataset.csv ==")
    run_one(ref, "REFERENCE(dataset.csv)", ref_out)

    # --- Your run (your computed descriptors)
    my = _strip_cols(pd.read_csv(MY_DATASET_CSV))
    # Drop ddGglyc columns from my_dataset.csv; we'll use the ground truth from reference
    my = my.drop(columns=["ddGglyc", "ddGglyc error"], errors="ignore")

    # merge ddGglyc from reference (so your descriptors get the GT labels)
    gt = ref[["name", "interactor", "ddGglyc", "ddGglyc error"]].copy()
    usr = gt.merge(my, on=["name", "interactor"], how="left")

    usr_out = OUT_DIR / "user"
    print("\n== Running user ML on my_dataset.csv (merged with ddGglyc from dataset.csv) ==")
    run_one(usr, "USER(my_dataset.csv)", usr_out)

    # diagnostics: how close are your descriptors to reference?
    diag_out = OUT_DIR / "diagnostics"
    print("\n== Writing descriptor difference diagnostics (reference vs user) ==")
    descriptor_diff_report(ref, usr, diag_out)

    print(f"\nDone. Results saved under: {OUT_DIR}")

if __name__ == "__main__":
    main()