# IT3091 Machine Learning — project record (history up to 21 September 2026)

> Restored on 2026-09-28 from the last version of `plan.md` (updated 17–21 September 2026), which was
> emptied so `plan.md` could hold the fresh to-do list. Links point one level up from `reports/`.
> The active plan is [`plan.md`](../plan.md). Section numbers are kept so older references still work.

**Group 2026-AI-46 · Lab Y3.S1.WD.AI.0101 · Guided Data Track, code 6: Tourism & Hospitality.**

> **Status, 21 September 2026 — read before quoting any number.** Sections 1–8 describe the **v1**
> foundation: deduplicated population, row-stratified 80/20 split, notebooks in `notebooks/v1/` and
> `reports/results/`. Those numbers are historical. **v2** (see `reports/training_foundation.md`) keeps
> the same population (84,969 rows, duplicates removed) and 25 predictors, and changes the split:
> rows with an identical predictor profile stay together. A rerun of notebooks 01 → `v2/02` → `v2/03`
> reproduced every manifest hash: `data/processed/v2/train.csv` (67,974 rows), `test.csv` (16,995),
> `cv_folds.csv` (5 outer × 3 inner group-aware folds, 0 shared profiles) and `manifest.json`. The v1
> split had 93 profiles on both sides; v2 has 0. v1 AP must not be compared with v2 AP. The local
> `archive/` folder is empty, so the archived material mentioned below is not available.
>
> **§9.3 done (21 September 2026).** Notebooks `v2/04`–`v2/09` retrained all six models with nested CV
> on the saved folds; `v2/10` compares them. Nested-CV AP, mean ± SD over 5 outer folds: Dummy 0.2775,
> Logistic Regression 0.6410 ± 0.0091, Decision Tree 0.7067 ± 0.0069, Random Forest 0.7552 ± 0.0082,
> **XGBoost 0.7644 ± 0.0080 (reference model)**, Neural Network 0.7437 ± 0.0101. Results in
> `reports/results/v2/` (per-model CSVs, `leaderboard.csv`, `oof/`).

## 1. Dataset and provenance

- [Hotel Booking Demand on Kaggle — Jesse Mostipak](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).
- [Original study: Antonio, Almeida and Nunes](https://doi.org/10.1016/j.dib.2018.11.126), *Data in Brief* 22 (2019), 41–49. [Open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/).
- Local file `data/raw/hotel_bookings.csv`, downloaded 2026-09-03 from the [TidyTuesday mirror](https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv).
- 16,855,599 bytes, 119,390 data rows, 32 columns. SHA-256 `7c2ae42a…c0c7b1fc06`.
- Kaggle names the requested dataset; the local artefact came from TidyTuesday. No byte-equality with Kaggle is claimed. Licence details in [SOURCE.md](../data/raw/SOURCE.md).

Two hotels in Portugal, arrivals July 2015 – August 2017, so 2015 and 2017 are partial years.

## 2. Problem, stakeholder and prediction time

The stakeholder is a hotel revenue manager. The target is `is_canceled` (1 cancelled, 0 not).
The model is intended to score a booking **at the moment it is made**, so every input must exist at
that moment. That constraint, not predictive strength, drove feature selection.

| Fields | Reason for exclusion |
| --- | --- |
| `reservation_status`, `reservation_status_date` | Record the outcome itself — direct leakage |
| `required_car_parking_spaces`, `assigned_room_type`, `booking_changes`, `days_in_waiting_list` | Filled in or updated after the booking is made |
| `company` | 94.3% empty; the presence flag was tested separately and did not help |
| `arrival_date_year` | 2015 covers 6 months and 2017 covers 8, so the apparent yearly trend is partly coverage; a deployed model would only ever see unseen years |
| `arrival_date_week_number` | Correlation 0.995 with month — one of the pair must go |

**What the exclusions cost, measured.** Restoring the four post-booking columns is worth
**+0.0407 PR-AUC** (0.8067 against the 0.7660 baseline, `reports/results/ablation.csv` arm 6). It was
declined, because a booking arriving today has no parking record, no room assignment and no change
history yet. `room_changed` (`reserved_room_type != assigned_room_type`) is worth **+0.0144** and is
excluded for the same reason (notebook 01 §7.2).

**Limitation.** The source study extracted values from booking logs relative to pre-arrival time
where available, so the CSV is not a verified snapshot of the booking-creation moment. `adr` and
`deposit_type` are retained on the argument that they are agreed at booking, but the file cannot prove
no later amendment is reflected. Claims are limited to **retrospective classification on held-out
historical bookings**.

## 3. EDA

Notebook 01 covers structure and types, numeric summaries, cardinality, missingness, impossible and
extreme values, duplicates, and cancellation patterns, each chart with a written reading. The
[insight log](eda_insight_log.md) records observation → evidence → explanation → consequence → uncertainty.

| lead time (days) | cancellation rate |
| --- | ---: |
| 0–30 | 16.5% |
| 91–180 | 35.3% |
| 366+ | 42.9% |

City Hotel cancels more than Resort Hotel (30.3% against 23.7%). Non Refund deposits cancel 94.9% of
the time but cover under 1% of rows. Online TA is the riskiest large segment at 35.3%. 88.5% of guests
are European; domestic Portuguese guests cancel at 35.9% against 22.5% for the rest of Europe.
These are associations, not causes.

**Outcome exploration re-checked after the split.** Notebook 02 §5.1 redraws the four charts that drove
a decision on training rows only. Every large group moves by under half a percentage point; only tiny
groups (`Refundable` −3.25, `Aviation` −2.27, `Undefined`) shift more than a point. No feature decision
depended on the test rows.

## 4. Cleaning

| Rule | Rows affected | Reason |
| --- | ---: | --- |
| `country` → `Unknown` | 488 | Own level; do not invent the modal origin |
| `agent` NaN → 0, kept as text | 16,340 | NaN means no travel agent; IDs are categories |
| `children` NaN → 0 | 4 | Four rows; stated as an approximation |
| `meal` Undefined → SC | 1,169 | The source dictionary defines them as the same |
| Drop `company` | — | 94.3% empty |
| Remove zero-guest rows | 180 | Not a bookable stay |
| Remove `adr` < 0 or ≥ 5000 | 2 | The two demonstrable errors only |
| Remove exact duplicate rows | 34,239 | See below |

119,390 → **84,969 bookings, 25 features**. No percentile-based outlier deletion: extreme is not invalid.

**Duplicates.** No booking ID exists, so a copy cannot be told apart from a genuine second booking.
Removing duplicates moved the cancellation rate from 37.1% to 27.7%; the population is **unique booking
profiles**. Keeping them scores **PR-AUC 0.9225 against 0.7660** (`ablation.csv` arm 7) — memorisation,
not skill. Limitations: identical rows may be legitimate separate bookings; and 93 test rows in v1 had
an identical profile in training (fixed in v2 by the group-aware split).

## 5. Feature engineering: measured, not assumed

Built: `total_nights` (weekday + weekend nights), `total_guests` (adults + children + babies),
`is_family` (children > 0 or babies > 0). Candidates from published notebooks, scored on the same folds
(`scripts/ablation.py`), baseline **PR-AUC 0.7660 ± 0.0071**:

| Candidate | Δ PR-AUC |
| --- | ---: |
| `prev_cancel_ratio` + `total_previous_bookings` | −0.0007 |
| `long_lead_time` | −0.0001 |
| `arrival_quarter` | +0.0018 |
| `booked_by_company` | −0.0001 |
| `adr_per_person` | −0.0008 |
| Frequency encoding for `country` and `agent` | −0.0010 |
| **All clean reference features together** | **+0.0026** |

None clears the 0.0071 noise floor. `is_family` contributes 0.0001 on permutation importance.

## 6. Preprocessing

**Encoding.** `OneHotEncoder(min_frequency=100, handle_unknown='infrequent_if_exist')`. 142 countries and
277 agents fall below 100 bookings (2.7% and 5.0% of rows) and share an infrequent bucket.
**Scaling.** `StandardScaler` for Logistic Regression and the neural network only.
**Nothing is fitted outside a fold** — encoder and scaler live inside each model's `Pipeline`.

**Class balance.** 27.7% positive. `class_weight='balanced'` / `scale_pos_weight` moved PR-AUC by
**−0.0100 to −0.0008, never positive** (`weight_sweep.csv`, `weighted_comparison.csv`,
`weighted_models.csv`). No SMOTE: the features are mostly categorical, so interpolation invents invalid
bookings, and resampling distorts the prevalence AP depends on. Imbalance is handled by the threshold.

**Feature ranking.** Permutation importance, training split only, five folds
(`scripts/feature_importance_check.py` → `feature_importance.csv`). Top: `country` 0.173,
`lead_time` 0.137, `agent` 0.128, `total_of_special_requests` 0.107. Automatic feature selection was not
used: it promotes `required_car_parking_spaces`, a post-booking leak.

## 7. Models and results (v1, historical — do not quote against v2)

| Model | Nested CV PR-AUC |
| --- | ---: |
| XGBoost | 0.7612 ± 0.0090 |
| Random Forest | 0.7548 ± 0.0079 |
| Neural Network | 0.7426 ± 0.0111 |
| Decision Tree | 0.7079 ± 0.0102 |
| Logistic Regression | 0.6470 ± 0.0095 |

v1 test pass (XGBoost, threshold 0.3352): PR-AUC 0.7698, ROC-AUC 0.8924, F1 0.7019, recall 0.7692,
precision 0.6454, accuracy 0.8187; TN 10,286 · FP 1,993 · FN 1,088 · TP 3,627.

## 8. Limitations to state in the report

1. **Test reuse (v1).** Earlier notebooks evaluated a test set of the same size; the v1 holdout was not
   untouched. v2 uses a new group-aware holdout, but the population was studied during EDA.
2. **93 overlapping profiles** in v1 (0 in v2).
3. **Threshold optimism** in v1 (threshold picked on OOF from hyperparameters chosen on all rows).
4. **No temporal validation.** Arrival date is not booking-creation date.
5. **No calibration assessment** yet.
6. **Scope.** Two Portuguese hotels, 2015–2017, pre-2020.
7. **F1-optimal is not cost-optimal.**

## 9. Why the ceiling (~0.76 AP) is not a modelling problem

- The three strongest models sit within about one fold SD of each other.
- Twelve feature candidates together added +0.0026, below the noise floor.
- Class weighting never helped.
- The only large gains were leakage: post-booking columns (+0.0407) and duplicates (+0.155).

The limit is the information available when the booking is made. The remaining marks come from
trustworthy evaluation and a cost-based decision, not from AP decimals.

### 9.1 Stop/go rule (applies to every experiment)

A change is adopted only if **all** hold:

1. Measured on **training data only**, on the frozen outer folds, as a *paired* per-fold difference
   against the current reference.
2. Mean paired AP gain exceeds **one outer-fold SD of the reference** (0.0080 for v2 XGBoost) and is
   positive in **≥ 4 of 5 folds** — or it lowers expected cost by more than the fold spread of cost.
3. Every input is known when the booking is made.
4. Extra fit time and complexity are stated and still worth it to a revenue manager.

### 9.3 v2 reference selection (21 September 2026)

Walk up the complexity order LR → DT → RF → XGBoost → NN (`reports/results/v2/reference_selection.csv`):

| Candidate | Against | Mean paired gain | Bar (reference SD) | Folds better | Adopted |
| --- | --- | --- | --- | --- | --- |
| Decision Tree | Logistic Regression | +0.0657 | 0.0091 | 5/5 | yes |
| Random Forest | Decision Tree | +0.0485 | 0.0069 | 5/5 | yes |
| XGBoost | Random Forest | +0.0092 | 0.0082 | 5/5 | yes |
| Neural Network | XGBoost | −0.0207 | 0.0080 | 0/5 | no |

**Reference: XGBoost** (600 rounds, depth 10, learning rate 0.05, chosen in every outer fold); about 30×
faster to refit than RF (6 s vs 191 s) and higher F1 at 0.5 (0.674 vs 0.631).

## 11. Contributions and AI use

The initial plan assigned M1 the shared foundation and Logistic Regression, M2 Decision Tree, M3 Random
Forest and M4 XGBoost. **In practice the project lead built the entire shared foundation alone** after
the schedule was compressed from seven weeks to three on 2026-09-03. The neural network is an addition
beyond the four planned models. Member names and per-model ownership must be confirmed before submission.

AI assistance was used to draft code and explanations, review the foundation, and run the ablation and
importance scripts. Reasoning and decisions were reviewed and accepted by the project lead.

Kaggle notebooks consulted for structure:

- [Hotel Booking Cancellation Prediction / EDA + 6 ML — Dalileh Oladzadeh](https://www.kaggle.com/code/dalileholadzadeh/hotel-booking-cancellation-prediction-eda-6-ml)
- [Hotel Cancellation Prediction using ANN — Aamir](https://www.kaggle.com/code/aamir5659/hotel-cancellation-prediction-using-ann)
- [HotelBookingCancellationAnalysis-V1 — Kushal](https://www.kaggle.com/code/kushal1147/hotelbookingcancellationanalysis-v1)
- [Hotel Booking — Hazem Alanany](https://www.kaggle.com/code/hazemalanany/hotel-booking)
- [Hotel Booking Cancellation Analysis / Python — Ahmed Baqa](https://www.kaggle.com/code/ahmedbaqa/hotel-booking-cancellation-analysis-python)

Their engineered features were measured (§5) rather than adopted on reputation (+0.0026 combined). Their
97–99% accuracy figures come from retaining outcome/post-booking columns, duplicates, or both.
