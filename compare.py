#!/usr/bin/env python3
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
OUT = BASE / "results_ml"
REF_DIR = OUT / "reference"
USR_DIR = OUT / "user"

DATASET = BASE / "dataset.csv"
MY_DATASET = BASE / "my_dataset.csv"

# Paper-reported targets (Fig 5)
PAPER_LOO_RMSE = 0.17
PAPER_OOB_RMSE = 0.17
PAPER_LOO_R2 = 0.74
PAPER_OOB_R2 = 0.75

TOL_RMSE = 0.03  # you can tighten/loosen
TOL_R2   = 0.08

CORE = [
    "PSAs+NSAi",
    "qNPSA",
    "HOMO-5s-LUMO+1i",  # may differ slightly in your CSV; we will fuzzy match
]

def load_metrics_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def rmse(y, yp):
    return float(math.sqrt(mean_squared_error(y, yp)))

def fuzzy_find_col(cols, target):
    # exact first
    if target in cols:
        return target
    # normalize
    def norm(s): return re.sub(r"[^a-z0-9]+", "", s.lower())
    t = norm(target)
    best = None
    best_score = 0
    for c in cols:
        cn = norm(c)
        # overlap heuristic
        score = 0
        if t in cn: score += 5
        if "homo" in t and "homo" in cn: score += 2
        if "lumo" in t and "lumo" in cn: score += 2
        if "psas" in t and "psas" in cn: score += 2
        if "qnpsa" in t and "qnpsa" in cn: score += 2
        if score > best_score:
            best_score = score
            best = c
    return best

def read_selected_features(dir_path: Path):
    p = dir_path / "selected_features.txt"
    if not p.exists():
        return []
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def get_user_table():
    ref = pd.read_csv(DATASET)
    my = pd.read_csv(MY_DATASET)
    # merge ddGglyc labels into your dataset
    my = my.drop(columns=["ddGglyc", "ddGglyc error"], errors="ignore")
    gt = ref[["name", "interactor", "ddGglyc", "ddGglyc error"]].copy()
    usr = gt.merge(my, on=["name", "interactor"], how="left")
    return usr

def fit_3feature_loocv(df, cols3):
    X = df[cols3].apply(pd.to_numeric, errors="coerce").values.astype(float)
    y = pd.to_numeric(df["ddGglyc"], errors="coerce").values.astype(float)

    # drop rows where y is nan
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LinearRegression()),
    ])
    loo = LeaveOneOut()
    yhat = cross_val_predict(pipe, X, y, cv=loo)
    # fit once on all data to get signs
    pipe.fit(X, y)
    coefs = pipe.named_steps["lr"].coef_.tolist()
    return {
        "loocv_rmse": rmse(y, yhat),
        "loocv_r2": float(r2_score(y, yhat)),
        "coef_signs": ["+" if c > 0 else "-" if c < 0 else "0" for c in coefs],
        "coefs": coefs,
    }

def main():
    report_dir = OUT / "compare"
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1) Does your selection contain the paper core features?
    user_selected = read_selected_features(USR_DIR)
    core_found = 0
    for k in CORE:
        if any(k == s for s in user_selected):
            core_found += 1

    # 2) Fit the 3-descriptor “paper-form” model on your data and check signs/perf
    usr = get_user_table()
    cols = list(usr.columns)

    c1 = fuzzy_find_col(cols, "PSAs+NSAi")
    c2 = fuzzy_find_col(cols, "qNPSA")
    c3 = fuzzy_find_col(cols, "HOMO-5s-LUMO+1i")

    three_cols = [c1, c2, c3]
    missing = [CORE[i] for i, c in enumerate(three_cols) if c is None]
    three_ok = all(c is not None for c in three_cols)

    three_fit = None
    sign_match = 0
    perf_match = 0

    if three_ok:
        three_fit = fit_3feature_loocv(usr, three_cols)
        # paper says all 3 coefficients are negative
        sign_match = sum(1 for s in three_fit["coef_signs"] if s == "-")

        # performance closeness
        if abs(three_fit["loocv_rmse"] - PAPER_LOO_RMSE) <= TOL_RMSE and abs(three_fit["loocv_r2"] - PAPER_LOO_R2) <= TOL_R2:
            perf_match += 1

    # 3) Compare your “USER” pipeline metrics vs paper targets (from your results_ml/user json)
    lin_m = load_metrics_json(USR_DIR / "linear_loocv_metrics.json") if (USR_DIR / "linear_loocv_metrics.json").exists() else None
    rf_m  = load_metrics_json(USR_DIR / "rf_oob_metrics.json") if (USR_DIR / "rf_oob_metrics.json").exists() else None

    perf_points = 0
    if lin_m:
        if abs(lin_m.get("rmse", 999) - PAPER_LOO_RMSE) <= TOL_RMSE:
            perf_points += 15
        if abs(lin_m.get("r2", -999) - PAPER_LOO_R2) <= TOL_R2:
            perf_points += 15
    if rf_m:
        if abs(rf_m.get("rmse", 999) - PAPER_OOB_RMSE) <= TOL_RMSE:
            perf_points += 15
        if abs(rf_m.get("r2", -999) - PAPER_OOB_R2) <= TOL_R2:
            perf_points += 15

    # 4) Final score
    # 50%: core features
    core_score = (core_found / 3) * 50
    # 20%: sign match (paper expects 3 negatives)
    sign_score = (sign_match / 3) * 20 if three_ok else 0
    # 30%: perf closeness using pipeline metrics (up to 60 points -> scaled to 30)
    perf_score = (perf_points / 60) * 30

    total = core_score + sign_score + perf_score

    out = {
        "paper_targets": {
            "linear_loocv_rmse": PAPER_LOO_RMSE,
            "linear_loocv_r2": PAPER_LOO_R2,
            "rf_oob_rmse": PAPER_OOB_RMSE,
            "rf_oob_r2": PAPER_OOB_R2,
        },
        "core_features_expected": CORE,
        "user_selected_features_count": len(user_selected),
        "core_features_found_in_user_selected": int(core_found),
        "three_feature_columns_used": three_cols,
        "three_feature_missing": missing,
        "three_feature_fit": three_fit,
        "user_pipeline_linear_metrics": lin_m,
        "user_pipeline_rf_metrics": rf_m,
        "score_breakdown": {
            "core_score_50": core_score,
            "sign_score_20": sign_score,
            "perf_score_30": perf_score,
            "total_100": total,
        },
    }

    (report_dir / "conclusion_agreement_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved:", report_dir / "conclusion_agreement_report.json")
    print("Conclusion match %:", round(total, 2))

if __name__ == "__main__":
    main()