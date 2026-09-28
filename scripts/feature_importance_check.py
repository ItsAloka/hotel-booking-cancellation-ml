"""Cross-check the 25 features we selected by hand against a model's own ranking.

Section 4 of 01_eda_cleaning.ipynb selects features on availability at prediction
time, not on score. That is deliberate: an importance ranking on the raw columns
puts required_car_parking_spaces at the top, and that column is written at
check-in. So this script never sees the post-booking columns. It ranks only the
25 that already passed the availability test, and answers a narrower question:
among features we are willing to deploy, is any one of them dead weight?

Permutation importance, fitted and measured inside each fold, on the training
split only - the test split is never loaded. Permuting the original column rather
than the encoded matrix keeps one score per feature instead of one per dummy.

    .venv/Scripts/python.exe scripts/feature_importance_check.py

Writes reports/results/feature_importance.csv.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, train_test_split

from ablation import SEED, load_clean, baseline_frame, make_pipe, preprocessor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "results" / "feature_importance.csv"

N_REPEATS = 5


def main():
    t0 = time.time()
    df = load_clean()
    X, y = baseline_frame(df), df["is_canceled"]
    print(f"clean rows: {len(df):,}   features ranked: {X.shape[1]}")

    # same split as every ablation arm, so the numbers sit beside ablation.csv
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    per_fold = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        t = time.time()
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipe = make_pipe()
        pipe.set_params(prep=preprocessor(X_tr))
        pipe.fit(X_tr, y_tr)                  # encoder fitted on this fold's train rows only

        # scoring on the held-out fold: importance is "what the model needs to
        # generalise", not "what it used to memorise the rows it was fitted on"
        r = permutation_importance(pipe, X_val, y_val, scoring="average_precision",
                                   n_repeats=N_REPEATS, random_state=SEED, n_jobs=-1)
        per_fold.append(pd.Series(r.importances_mean, index=X.columns, name=f"fold_{fold}"))
        print(f"  fold {fold} done ({time.time() - t:.0f} s)")

    folds = pd.concat(per_fold, axis=1)
    out = pd.DataFrame({
        "feature": folds.index,
        "ap_drop_mean": folds.mean(axis=1),
        "ap_drop_std": folds.std(axis=1),
        "folds_positive": (folds > 0).sum(axis=1),   # 5/5 means every fold agreed it helps
    }).sort_values("ap_drop_mean", ascending=False).reset_index(drop=True)

    # a feature has to help in every fold and move AP by more than the ablation
    # noise floor before we would call it load-bearing; below that it is a
    # candidate to question, NOT an instruction to drop - correlated features
    # share credit and each looks small when the other is still present.
    out["verdict"] = np.where(
        (out["ap_drop_mean"] > 0.0071) & (out["folds_positive"] == 5), "load-bearing",
        np.where(out["ap_drop_mean"] <= 0.0005, "question it", "minor"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.round(5).to_csv(OUT, index=False)

    print(f"\n{'=' * 78}")
    print(out.round(5).to_string(index=False))
    print("\nPermuted on the held-out fold, training split only. Post-booking columns")
    print("were never candidates here - section 4 excluded them before this ran.")
    print(f"written to {OUT.relative_to(ROOT)}   total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
