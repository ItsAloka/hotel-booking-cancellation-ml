"""Does class weighting help? Every model, weighted against unweighted, same folds.

27.7% of bookings are cancelled. That is imbalanced enough to ask the question and
mild enough that the answer is not obvious, so it is measured rather than assumed.

Both arms use each model's already-tuned hyperparameters from reports/results/, so
the only difference between a pair of rows is the weighting. That makes the pair
comparable to each other - but NOT to the nested CV numbers in the leaderboard,
which include the cost of searching for those hyperparameters. The unweighted arm
here is re-run for exactly that reason: it is the like-for-like partner.

Training data only. The test set is not opened - the model choice is already frozen.

    .venv/Scripts/python.exe scripts/weighted_comparison.py

Writes reports/results/weighted_<model>.csv and weighted_comparison.csv.
"""

import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"
SEED = 42


class OversampledMLP(BaseEstimator, ClassifierMixin):
    """MLPClassifier with class weighting faked by replicating minority rows.

    scikit-learn's MLPClassifier accepts neither class_weight nor sample_weight, so
    the weighted arm cannot be expressed the way it is for the other four models.
    Duplicating existing minority rows until the classes balance is the closest
    honest equivalent: it reweights the loss without inventing a single booking, so
    it is not SMOTE and creates no synthetic feature combinations. Resampling happens
    inside fit, which means inside each fold - never before the split.
    """

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        rng = np.random.RandomState(SEED)
        y = np.asarray(y)
        minority = int(pd.Series(y).value_counts().idxmin())
        minority_idx = np.flatnonzero(y == minority)
        n_needed = int((y != minority).sum() - (y == minority).sum())
        extra = rng.choice(minority_idx, size=n_needed, replace=True)
        idx = np.concatenate([np.arange(len(y)), extra])
        rng.shuffle(idx)

        self.model_ = MLPClassifier(**self.params).fit(X[idx], y[idx])
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


def parse_params(raw):
    """best_params is stored as JSON with everything stringified; put the types back."""
    out = {}
    for key, value in json.loads(raw).items():
        try:
            out[key.replace("model__", "")] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[key.replace("model__", "")] = value
    return out


def build(name, params, weighted):
    """The estimator plus whether its numeric block needs scaling."""
    if name == "Logistic Regression":
        kw = {"class_weight": "balanced"} if weighted else {}
        return LogisticRegression(max_iter=1000, random_state=SEED, **params, **kw), True
    if name == "Decision Tree":
        kw = {"class_weight": "balanced"} if weighted else {}
        return DecisionTreeClassifier(random_state=SEED, **params, **kw), False
    if name == "Random Forest":
        kw = {"class_weight": "balanced"} if weighted else {}
        return RandomForestClassifier(random_state=SEED, n_jobs=-1, **params, **kw), False
    if name == "XGBoost":
        # xgboost has no class_weight; scale_pos_weight = negatives / positives is the equivalent
        kw = {"scale_pos_weight": SCALE_POS} if weighted else {}
        return XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss",
                             importance_type="gain", **params, **kw), False
    if name == "Neural Network":
        base = dict(max_iter=500, early_stopping=True, random_state=SEED, **params)
        return (OversampledMLP(**base) if weighted else MLPClassifier(**base)), True
    raise ValueError(name)


def make_pipe(estimator, needs_scaling, categorical, numeric):
    prep = ColumnTransformer([
        ("num", StandardScaler() if needs_scaling else "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100), categorical),
    ], verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("model", estimator)])


def score_arm(pipe, X, y, cv):
    """Outer-fold AP/ROC, plus a threshold chosen from out-of-fold probabilities."""
    t = time.time()
    ap = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision", n_jobs=1)
    roc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y, oof)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(np.argmax(f1[:-1]))

    return {
        "cv_pr_auc_mean": ap.mean(), "cv_pr_auc_std": ap.std(), "cv_roc_auc_mean": roc.mean(),
        "oof_pr_auc": average_precision_score(y, oof),
        "f1_at_default_0.50": f1_score(y, oof >= 0.5),
        "threshold": float(thresholds[best]),
        "f1_at_threshold": float(f1[best]),
        "recall_at_threshold": float(recall[best]),
        "precision_at_threshold": float(precision[best]),
        "seconds": time.time() - t,
        "fold_scores": list(np.round(ap, 4)),
    }


def main():
    global SCALE_POS
    t0 = time.time()

    train = pd.read_csv(ROOT / "data/processed/train.csv", dtype={"agent": str})
    X, y = train.drop(columns=["is_canceled"]), train["is_canceled"]
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    numeric = X.select_dtypes(include="number").columns.tolist()

    SCALE_POS = float((y == 0).sum() / (y == 1).sum())
    print(f"training rows: {len(X):,}   cancellation rate: {y.mean():.4f}   "
          f"scale_pos_weight would be {SCALE_POS:.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    slugs = {"Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree",
             "Random Forest": "random_forest", "XGBoost": "xgboost",
             "Neural Network": "neural_network"}

    rows = []
    for name, slug in slugs.items():
        params = parse_params(pd.read_csv(RESULTS / f"{slug}.csv")["best_params"].iloc[0])
        print(f"\n{name}  {params}")

        for weighted in (False, True):
            estimator, needs_scaling = build(name, params, weighted)
            pipe = make_pipe(estimator, needs_scaling, categorical, numeric)
            result = score_arm(pipe, X, y, cv)
            label = "balanced" if weighted else "none"
            if weighted and name == "Neural Network":
                label = "balanced (by replication)"
            rows.append({"model": name, "weighting": label, **result})
            print(f"  {label:<26} AP {result['cv_pr_auc_mean']:.4f} +/- {result['cv_pr_auc_std']:.4f}"
                  f"   F1@0.50 {result['f1_at_default_0.50']:.4f}"
                  f"   F1@thr {result['f1_at_threshold']:.4f}  ({result['seconds']:.0f} s)")

    out = pd.DataFrame(rows)

    # the paired difference: weighted minus unweighted, same model, same folds
    pivot = out.pivot_table(index="model", columns="weighting", values="cv_pr_auc_mean")
    weighted_col = pivot.columns.difference(["none"])
    pivot["delta_ap"] = pivot[weighted_col].max(axis=1) - pivot["none"]
    out = out.merge(pivot[["delta_ap"]], on="model", how="left")
    out.loc[out["weighting"] == "none", "delta_ap"] = 0.0

    for name, slug in slugs.items():
        out[out["model"] == name].round(4).to_csv(RESULTS / f"weighted_{slug}.csv", index=False)
    out.round(4).to_csv(RESULTS / "weighted_comparison.csv", index=False)

    print(f"\n{'=' * 104}")
    show = ["model", "weighting", "cv_pr_auc_mean", "cv_pr_auc_std", "delta_ap",
            "f1_at_default_0.50", "threshold", "f1_at_threshold", "recall_at_threshold"]
    print(out[show].round(4).to_string(index=False))
    print(f"\nwritten to {RESULTS.relative_to(ROOT)}/weighted_*.csv   total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
