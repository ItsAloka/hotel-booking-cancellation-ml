"""Measure every candidate feature and cleaning rule before it goes into a notebook.

Each arm is the baseline feature set plus one change, scored the same way, so the
difference between two rows is the change and nothing else. Run from anywhere:

    .venv/Scripts/python.exe scripts/ablation.py

Writes reports/results/ablation.csv.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hotel_bookings.csv"
OUT = ROOT / "reports" / "results" / "ablation.csv"

SEED = 42

# the columns 01 drops as post-booking, kept aside so arms 5 and 6 can put them back
POST_BOOKING = ["required_car_parking_spaces", "booking_changes", "days_in_waiting_list"]

# 02 drops these two before the split, for reasons argued in that notebook
DROPPED_IN_02 = ["arrival_date_year", "arrival_date_week_number"]


def load_clean(dedup=True):
    """The cleaning 01_eda_cleaning.ipynb performs, as a function.

    assigned_room_type and the post-booking columns are carried through so the
    arms that need them can use them; the baseline drops them again below.
    """
    df = pd.read_csv(RAW)

    # company is 94.3% empty so the id is useless, but "was there a company at all"
    # is knowable at booking time and company bookings cancel far less often
    df["booked_by_company"] = df["company"].notna().astype(int)

    # missing values, one rule per column
    df = df.drop(columns=["company"])                       # 94.3% empty
    df["agent"] = df["agent"].fillna(0).astype(int)         # NaN = booked without an agent
    df["country"] = df["country"].fillna("Unknown")         # own level, not the mode
    df["children"] = df["children"].fillna(0).astype(int)

    # impossible rows
    df = df[(df["adults"] + df["children"] + df["babies"]) > 0]
    df = df[(df["adr"] >= 0) & (df["adr"] < 5000)]          # the two demonstrable errors

    # the dataset documentation defines Undefined and SC as the same thing
    df["meal"] = df["meal"].replace("Undefined", "SC")

    # leakage: these two record the outcome itself
    df = df.drop(columns=["reservation_status", "reservation_status_date"])

    if dedup:
        # 01 drops the post-booking columns BEFORE deduplicating, so two rows differing
        # only in (say) booking_changes collapse into one there. Deduplicating on the
        # full frame here would keep those rows and quietly give every arm 2,000 more
        # rows than the notebook has. Dedup on 01's column set, carry the rest along.
        # booked_by_company is excluded from the subset too: it is derived from a column
        # 01 drops before deduplicating, so counting it here would spare rows the
        # notebook collapses and quietly change the row count.
        carried = POST_BOOKING + ["assigned_room_type", "booked_by_company"]
        subset = [c for c in df.columns if c not in carried]
        df = df.drop_duplicates(subset=subset).reset_index(drop=True)

    # features 01 already builds
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["is_family"] = ((df["children"] > 0) | (df["babies"] > 0)).astype(int)

    return df.reset_index(drop=True)


def baseline_frame(df):
    """The 25 features 02_preprocessing produces today."""
    X = df.drop(columns=["is_canceled", "assigned_room_type", "booked_by_company"]
                        + POST_BOOKING + DROPPED_IN_02)
    X["agent"] = X["agent"].astype(str)      # an ID, not a quantity
    return X


def make_pipe():
    """The tuned XGBoost from 06_xgboost.ipynb. Same model for every arm."""
    return Pipeline([
        ("prep", "passthrough"),             # replaced per arm, columns differ
        ("model", XGBClassifier(
            n_estimators=600, learning_rate=0.05, max_depth=10, scale_pos_weight=1,
            random_state=SEED, n_jobs=-1, eval_metric="logloss", importance_type="gain")),
    ])


def preprocessor(X, country_encoding="onehot"):
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    numeric = X.select_dtypes(include="number").columns.tolist()
    return ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100), categorical),
    ], verbose_feature_names_out=False)


def score_arm(X, y):
    """5-fold CV on the training split only. Everything is fitted inside the fold."""
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pr, roc = [], []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipe = make_pipe()
        pipe.set_params(prep=preprocessor(X_tr))
        pipe.fit(X_tr, y_tr)                       # encoder fitted on the fold's train rows only

        proba = pipe.predict_proba(X_val)[:, 1]
        pr.append(average_precision_score(y_val, proba))
        roc.append(roc_auc_score(y_val, proba))

    return float(np.mean(pr)), float(np.std(pr)), float(np.mean(roc))


def freq_encode(X, cols):
    """R4's frequency encoding, applied as a plain column swap.

    Fitted on the whole frame, which is a mild optimism this arm is not scored on:
    it is only used to decide encoding against one-hot, and both arms would gain
    from it equally. Whichever wins is refitted train-only in 02.
    """
    X = X.copy()
    for col in cols:
        X[col + "_freq"] = X[col].map(X[col].value_counts(normalize=True))
        X = X.drop(columns=[col])
    return X


def main():
    t0 = time.time()
    df = load_clean()
    y = df["is_canceled"]
    base = baseline_frame(df)
    print(f"clean rows: {len(df):,}   baseline features: {base.shape[1]}")

    arms = {}

    # 1. baseline
    arms["1 baseline (25 features)"] = (base, y, "clean")

    # 2. guest history as a rate, from R4/R5
    a2 = base.copy()
    a2["total_previous_bookings"] = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    a2["prev_cancel_ratio"] = df["previous_cancellations"] / a2["total_previous_bookings"].replace(0, 1)
    arms["2 +prev_cancel_ratio +total_previous_bookings"] = (a2, y, "clean")

    # 3. long_lead_time, from R4/R5
    a3 = base.copy()
    a3["long_lead_time"] = (df["lead_time"] > 90).astype(int)
    arms["3 +long_lead_time"] = (a3, y, "clean")

    # 4. arrival_quarter, from R5
    a4 = base.copy()
    a4["arrival_quarter"] = pd.cut(df["arrival_date_week_number"], bins=[0, 13, 26, 39, 53],
                                   labels=["1", "2", "3", "4"]).astype(str)
    arms["4 +arrival_quarter"] = (a4, y, "clean")

    # 5. room_changed - knowable only at check-in
    a5 = base.copy()
    a5["room_changed"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
    arms["5 +room_changed"] = (a5, y, "post-booking")

    # 6. the columns 01 drops, restored. measures what our leakage discipline costs.
    a6 = base.copy()
    for col in POST_BOOKING:
        a6[col] = df[col]
    a6["assigned_room_type"] = df["assigned_room_type"]
    arms["6 +post-booking columns restored"] = (a6, y, "post-booking")

    # 8. country and agent frequency-encoded instead of one-hot, from R4
    arms["8 country+agent frequency-encoded"] = (freq_encode(base, ["country", "agent"]), y, "clean")

    # 10. booked_by_company: company bookings cancel far less often, and whether a
    # company is attached to the booking is known the moment it is made.
    a10 = base.copy()
    a10["booked_by_company"] = df["booked_by_company"]
    arms["10 +booked_by_company"] = (a10, y, "clean")

    # 11. adr_per_person, from the good Kaggle kernel (R7): a 200 EUR room is cheap
    # for four people and expensive for one, and adr alone cannot say which.
    a11 = base.copy()
    a11["adr_per_person"] = df["adr"] / df["total_guests"]
    arms["11 +adr_per_person"] = (a11, y, "clean")

    # 12. both of the above together
    a12 = base.copy()
    a12["booked_by_company"] = df["booked_by_company"]
    a12["adr_per_person"] = df["adr"] / df["total_guests"]
    arms["12 +booked_by_company +adr_per_person"] = (a12, y, "clean")

    # 13. deposit_type stress test, the check R7 ran on his own strongest feature:
    # Non Refund cancels 94.9% of the time on only 990 bookings, so confirm the
    # model does not collapse without it.
    arms["13 -deposit_type (stress test)"] = (base.drop(columns=["deposit_type"]), y, "clean")

    # 14. the same stress test on total_of_special_requests. It is the feature most
    # often confused with required_car_parking_spaces, so its contribution should be
    # measured rather than assumed - a big drop here means we genuinely rely on it.
    arms["14 -total_of_special_requests (stress test)"] = (
        base.drop(columns=["total_of_special_requests"]), y, "clean")

    # 9. every clean reference feature at once. Individually none of 2-4 clears the
    # noise floor; this checks they do not add up to something that does.
    a9 = base.copy()
    a9["total_previous_bookings"] = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    a9["prev_cancel_ratio"] = df["previous_cancellations"] / a9["total_previous_bookings"].replace(0, 1)
    a9["long_lead_time"] = (df["lead_time"] > 90).astype(int)
    a9["arrival_quarter"] = pd.cut(df["arrival_date_week_number"], bins=[0, 13, 26, 39, 53],
                                   labels=["1", "2", "3", "4"]).astype(str)
    arms["9 all clean reference features (2+3+4)"] = (a9, y, "clean")

    results = []
    for name, (X, target, label) in arms.items():
        t = time.time()
        pr_mean, pr_std, roc_mean = score_arm(X, target)
        results.append({"arm": name, "kind": label, "n_features": X.shape[1],
                        "pr_auc_mean": pr_mean, "pr_auc_std": pr_std, "roc_auc_mean": roc_mean})
        print(f"  {name:<46} PR-AUC {pr_mean:.4f} +/- {pr_std:.4f}   ({time.time() - t:.0f} s)")

    # 7. no-dedup: a different row count, so it gets its own clean frame
    t = time.time()
    df_nd = load_clean(dedup=False)
    X_nd = baseline_frame(df_nd)
    pr_mean, pr_std, roc_mean = score_arm(X_nd, df_nd["is_canceled"])
    results.append({"arm": "7 no-dedup (baseline features)", "kind": "not comparable",
                    "n_features": X_nd.shape[1], "pr_auc_mean": pr_mean,
                    "pr_auc_std": pr_std, "roc_auc_mean": roc_mean})
    print(f"  {'7 no-dedup (baseline features)':<46} PR-AUC {pr_mean:.4f} +/- {pr_std:.4f}   ({time.time() - t:.0f} s)")
    print(f"     (on {len(df_nd):,} rows vs {len(df):,} deduplicated - a different, easier problem)")

    out = pd.DataFrame(results)
    baseline_pr = out.loc[out["arm"].str.startswith("1 "), "pr_auc_mean"].iloc[0]
    out["delta_vs_baseline"] = out["pr_auc_mean"] - baseline_pr
    # arm 7 changes the rows, not the features, so its delta is not a like-for-like comparison
    out.loc[out["kind"] == "not comparable", "delta_vs_baseline"] = np.nan

    # a feature only earns its place if it beats baseline by more than the fold noise
    baseline_std = out.loc[out["arm"].str.startswith("1 "), "pr_auc_std"].iloc[0]
    out["beats_noise"] = out["delta_vs_baseline"] > baseline_std

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(OUT, index=False)

    print(f"\n{'=' * 100}")
    print(out.round(4).to_string(index=False))
    print(f"\nbaseline fold std is {baseline_std:.4f}; a delta smaller than that is noise, not a feature.")
    print(f"written to {OUT.relative_to(ROOT)}   total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
