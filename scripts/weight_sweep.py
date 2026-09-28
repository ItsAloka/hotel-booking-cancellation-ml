"""How much weight? Sweeping the minority-class weight instead of testing one value.

Notebook 09 tests class weighting on or off, where "on" means the `balanced` value -
2.604 for this dataset, the ratio that makes both classes contribute equally to the
loss. That is a formula, not a chosen optimum, so this sweeps the weight instead:
1 (none), 2.604 (balanced), 5 and 10.

Hyperparameters are held at each model's tuned values from reports/results/, so the
only thing changing down a column is the weight. Two models: XGBoost because it is
the model selected in 08, Logistic Regression because the effect is largest there.

One pass of the five outer folds per arm, refitting inside each fold. Per-fold AP and
the out-of-fold probabilities come from the same five fits rather than from separate
cross_val_score and cross_val_predict calls, which halves the runtime.

Training data only - the test set is not opened.

    .venv/Scripts/python.exe scripts/weight_sweep.py

Writes reports/results/weight_sweep.csv.
"""

import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"
SEED = 42


def tuned_params(slug):
    """The hyperparameters 03-07 settled on, with their types put back."""
    raw = pd.read_csv(RESULTS / f"{slug}.csv")["best_params"].iloc[0]
    out = {}
    for key, value in json.loads(raw).items():
        try:
            out[key.replace("model__", "")] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[key.replace("model__", "")] = value
    return out


def build(name, params, weight):
    """weight is the multiplier on the cancelled class; 1.0 means no weighting."""
    if name == "XGBoost":
        return XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss",
                             importance_type="gain", scale_pos_weight=weight, **params), False
    class_weight = None if weight == 1.0 else {0: 1.0, 1: weight}
    return LogisticRegression(max_iter=1000, random_state=SEED,
                              class_weight=class_weight, **params), True


def run_arm(estimator, needs_scaling, X, y, categorical, numeric, cv):
    """Five fits: per-fold AP and the full out-of-fold probability vector from one pass."""
    oof = np.zeros(len(y))
    fold_ap = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        prep = ColumnTransformer([
            ("num", StandardScaler() if needs_scaling else "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100), categorical),
        ], verbose_feature_names_out=False)

        pipe = Pipeline([("prep", prep), ("model", estimator)]).fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_val)[:, 1]
        oof[val_idx] = proba
        fold_ap.append(average_precision_score(y_val, proba))

    precision, recall, thresholds = precision_recall_curve(y, oof)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(np.argmax(f1[:-1]))

    return {
        "cv_pr_auc_mean": float(np.mean(fold_ap)),
        "cv_pr_auc_std": float(np.std(fold_ap)),
        "oof_pr_auc": average_precision_score(y, oof),
        "oof_roc_auc": roc_auc_score(y, oof),
        "f1_at_0.50": f1_score(y, oof >= 0.5),
        "threshold": float(thresholds[best]),
        "f1_at_threshold": float(f1[best]),
        "recall_at_threshold": float(recall[best]),
        "precision_at_threshold": float(precision[best]),
        "fold_scores": json.dumps([round(s, 4) for s in fold_ap]),
    }


def main():
    t0 = time.time()

    train = pd.read_csv(ROOT / "data/processed/train.csv", dtype={"agent": str})
    X, y = train.drop(columns=["is_canceled"]), train["is_canceled"]
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    numeric = X.select_dtypes(include="number").columns.tolist()

    balanced = float((y == 0).sum() / (y == 1).sum())
    weights = [1.0, round(balanced, 3), 5.0, 10.0]
    print(f"training rows: {len(X):,}   cancelled: {y.mean():.2%}   balanced weight = {balanced:.3f}")
    print(f"sweeping weights: {weights}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []

    for name, slug in [("XGBoost", "xgboost"), ("Logistic Regression", "logistic_regression")]:
        params = tuned_params(slug)
        print(f"--- {name}  {params}")

        for weight in weights:
            t = time.time()
            estimator, needs_scaling = build(name, params, weight)
            result = run_arm(estimator, needs_scaling, X, y, categorical, numeric, cv)
            note = {1.0: "none", round(balanced, 3): "balanced"}.get(weight, "")
            rows.append({"model": name, "weight": weight, "note": note, **result,
                         "seconds": time.time() - t})
            print(f"    weight {weight:<6} {note:<9} AP {result['cv_pr_auc_mean']:.4f} "
                  f"+/- {result['cv_pr_auc_std']:.4f}   thr {result['threshold']:.4f}   "
                  f"F1@thr {result['f1_at_threshold']:.4f}   "
                  f"recall {result['recall_at_threshold']:.4f}   ({time.time() - t:.0f} s)")
        print()

    out = pd.DataFrame(rows)

    # every arm compared with its own model's unweighted arm
    base = out[out["weight"] == 1.0].set_index("model")
    out["delta_ap"] = out.apply(
        lambda r: r["cv_pr_auc_mean"] - base.loc[r["model"], "cv_pr_auc_mean"], axis=1)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(RESULTS / "weight_sweep.csv", index=False)

    print("=" * 104)
    show = ["model", "weight", "note", "cv_pr_auc_mean", "delta_ap", "f1_at_0.50",
            "threshold", "f1_at_threshold", "recall_at_threshold", "precision_at_threshold"]
    print(out[show].round(4).to_string(index=False))
    print(f"\nwritten to {(RESULTS / 'weight_sweep.csv').relative_to(ROOT)}   "
          f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
