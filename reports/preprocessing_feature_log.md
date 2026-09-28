# Preprocessing and feature log

For each data issue: the options we considered, what we chose, and the evidence.
Cleaning is in `notebooks/01_eda_cleaning.ipynb`, encoding and the split in `02_preprocessing.ipynb`,
the class-weighting check in `09_class_weighting.ipynb`. Feature tests were run with `scripts/ablation.py`
(results in `reports/results/ablation.csv`): each test adds or removes something and scores XGBoost on the
same 5 training folds.

**Noise rule.** The baseline scores **PR-AUC 0.7660 ± 0.0071** across the 5 folds. A change only counts if
it moves the score by more than that ±0.0071 "wobble" between folds.

## Rubric items

| Issue | Options considered | What we chose | Evidence |
| --- | --- | --- | --- |
| **Missing values** | (a) drop rows, (b) fill everything with the most common value, (c) a separate rule per column | **(c)**: `company` dropped (94.3% empty); `agent` NaN → 0 = "no agent" (16,340 rows); `country` NaN → `Unknown` (488); `children` NaN → 0 (4); `meal` "Undefined" → "SC" (1,169, the dataset defines them as the same) | Missing agent = booked without one (mostly Direct/Corporate). Missing-country rows cancel 13.7%, but the mode PRT cancels 56.6%, so filling with PRT would mislabel them (01 §2) |
| **Outliers / impossible values** | (a) cut top/bottom 1% (percentile), (b) remove only values that cannot be real | **(b)**: removed 180 zero-guest bookings and 2 impossible prices (−6.38 and 5,400) | Next-highest price is 510, so 5,400 is an error. A percentile cut would delete real peak-season bookings. "Extreme" is not "wrong" (01 §3) |
| **Duplicates** | (a) keep, (b) remove exact duplicate rows | **(b)**: removed 34,239 rows → **84,969 bookings** | Cancel rate 37.1% → 27.7%. With duplicates kept, PR-AUC jumps to **0.9225** (vs 0.7660) because copies land in both train and test and the model memorises them (`ablation.csv` arm 7). Limitation: with no booking ID, some copies may be real separate bookings |
| **Categories** | (a) one-hot every level, (b) frequency encoding (replace a country by how common it is), (c) one-hot with rare levels grouped | **(c)**: `OneHotEncoder(min_frequency=100, handle_unknown='infrequent_if_exist')`. Levels with fewer than 100 bookings share one "infrequent" column. `agent` is treated as text, since agent 240 is not "bigger" than agent 9 | 142 countries and 277 agents have < 100 bookings but cover only 2.7% and 5.0% of rows (02 §3). Frequency encoding scored **−0.0010** (`ablation.csv` arm 8). One-hot is also easier to read ("country_PRT") |
| **Unseen categories** | (a) crash, (b) silently fill with zeros, (c) send to the "infrequent" column | **(c)**, via `handle_unknown='infrequent_if_exist'` | Happened in our own split: the `Undefined` market segment appears only in the test set (02 §5.1) |
| **Scaling** | (a) scale for all models, (b) scale only where it matters | **(b)**: `StandardScaler` for Logistic Regression and the Neural Network only | `lead_time` goes up to 737 while `is_family` is 0/1; linear models and neural networks are sensitive to this, tree models are not (they split on thresholds) (02 §3) |
| **Fitting on the right data** | Fit encoder/scaler on the whole table, or only on training data | Encoder and scaler sit **inside each model's `Pipeline`**, so they learn only from training rows | Otherwise test rows would influence the encoding before evaluation (02 §3) |
| **Class imbalance** | (a) do nothing, (b) class weighting (`class_weight='balanced'`, XGBoost `scale_pos_weight`), (c) SMOTE (make synthetic cancellations), (d) move the decision threshold | **(d)** plus a stratified split. Weighting tested and **rejected**; SMOTE **rejected** | Weighting changed PR-AUC by **−0.0100 to −0.0008 and never helped** (notebook 09, `weighted_comparison.csv`, `weight_sweep.csv`). SMOTE mixes two bookings to invent new ones; with mostly categorical features this creates bookings that cannot exist. 27.7% is not a starved class (18,861 cancellations in training). Choosing the threshold (0.335 for XGBoost) instead of 0.5 raised out-of-fold F1 from 0.673 to 0.698 (`leaderboard.csv`) |
| **Leakage (outcome columns)** | Keep or drop `reservation_status`, `reservation_status_date` | **Dropped** | `reservation_status` is the target written again: every "Canceled"/"No-Show" is a cancellation (01 §4) |
| **Leakage (post-booking columns)** | Keep or drop `required_car_parking_spaces`, `assigned_room_type`, `booking_changes`, `days_in_waiting_list` | **Dropped** | Filled in or changed after the booking is made. 7,409 bookings with parking and **0** cancelled: parking means "the guest arrived" (01 §4.1). Putting all four back would add **+0.0407** PR-AUC (0.8067, `ablation.csv` arm 6), but that gain is fake at booking time |
| **Other dropped columns** | Keep or drop | Dropped `arrival_date_year` (2015 and 2017 are partial years; future years are unseen) and `arrival_date_week_number` (0.995 correlated with month) | 02 §2 |
| **Train/test split** | Other ratios; unstratified | **80/20, stratified, seed 42**: 67,975 train / 16,994 test, both 27.75% cancelled | Stratification keeps the cancel rate equal on both sides (02 §5) |

## New features

| Feature | How it is built | Kept? | Evidence |
| --- | --- | --- | --- |
| `total_nights` | weekend nights + week nights | **Kept** | Simple, readable length of stay; permutation importance 0.008 (`feature_importance.csv`) |
| `total_guests` | adults + children + babies | **Kept** | Party size in one number; also used to find zero-guest rows |
| `is_family` | 1 if children > 0 or babies > 0 | **Kept** | Families cancel 34.2% vs 27.0% (01 §7). Adds almost nothing on top of the other columns (importance 0.0001, `feature_importance.csv`), but does no harm |
| `room_changed` | reserved room ≠ assigned room | **Rejected** | Works (+0.0144, `ablation.csv` arm 5) but uses `assigned_room_type`, which is only known at check-in: a leak (01 §7.2) |

## Kaggle notebook features that did not help

We tried the extra features used in popular Kaggle notebooks for this dataset. None beat the ±0.0071 noise
(`ablation.csv`; 01 §7.1):

| Candidate feature | Change in PR-AUC |
| --- | ---: |
| `prev_cancel_ratio` + `total_previous_bookings` | −0.0007 |
| `long_lead_time` (lead time > 90 days) | −0.0001 |
| `arrival_quarter` | +0.0018 |
| `booked_by_company` | −0.0001 |
| `adr_per_person` | −0.0008 |
| `booked_by_company` + `adr_per_person` | −0.0011 |
| Frequency encoding for `country` and `agent` | −0.0010 |
| **First three together** | **+0.0026** |

**Why they do not help:** they repeat what the model already knows from `lead_time`, `previous_cancellations`,
`arrival_date_month` and `market_segment`. So none were added. The Kaggle notebooks' 97–99% accuracy comes
from keeping leak columns and/or duplicates, not from these features.

## Feature selection

We did **not** use automatic feature selection: it promotes `required_car_parking_spaces`, which is a leak.
All 25 features go to every model. Permutation importance (shuffle one column, see how much the score drops),
training data only, 5 folds (`feature_importance.csv`): `country` 0.173, `lead_time` 0.137, `agent` 0.128,
`total_of_special_requests` 0.107.

**Final input table:** 84,969 bookings, 25 features (15 numeric, 10 categorical) + the target.
