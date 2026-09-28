# IT3091 Machine Learning — project record

**Group 2026-AI-46 · Lab Y3.S1.WD.AI.0101 · Guided Data Track, code 6: Tourism & Hospitality.**

This file collects the project's numbers and decisions in one place. Every number here comes from the
notebooks in `notebooks/` (01 to 10) and the CSVs in `reports/results/`.

## 1. Dataset and provenance

- [Hotel Booking Demand on Kaggle — Jesse Mostipak](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).
- [Original study: Antonio, Almeida and Nunes](https://doi.org/10.1016/j.dib.2018.11.126), *Data in Brief* 22 (2019), 41–49. [Open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/).
- Local file `data/raw/hotel_bookings.csv`, downloaded 2026-09-03 from the [TidyTuesday mirror](https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv).
- 16,855,599 bytes, 119,390 data rows, 32 columns. SHA-256 `7c2ae42a…c0c7b1fc06`.
- Kaggle names the requested dataset; the local artefact came from TidyTuesday. No byte-equality with Kaggle is claimed. Licence details in [SOURCE.md](../data/raw/SOURCE.md).

Two hotels in Portugal, arrivals July 2015 – August 2017, so 2015 and 2017 are partial years.

## 2. Problem, stakeholder and prediction time

The stakeholder is a hotel revenue manager. The target is `is_canceled` (1 cancelled, 0 not).
The model is meant to score a booking **at the moment it is made**, so every input must exist at
that moment. That rule, not predictive strength, decided which columns were kept.

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
`deposit_type` are kept on the argument that they are agreed at booking, but the file cannot prove
no later change is reflected. Claims are limited to **predicting cancellations on held-out
historical bookings**.

## 3. EDA

Notebook 01 covers structure and types, numeric summaries, cardinality, missingness, impossible and
extreme values, duplicates, and cancellation patterns, each chart with a written reading.

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

## 4. Cleaning (notebook 01)

| Rule | Rows affected | Reason |
| --- | ---: | --- |
| `country` → `Unknown` | 488 | Own level; do not invent the most common country |
| `agent` NaN → 0, kept as text | 16,340 | NaN means no travel agent; IDs are categories |
| `children` NaN → 0 | 4 | Four rows; stated as an approximation |
| `meal` Undefined → SC | 1,169 | The source dictionary defines them as the same |
| Drop `company` | — | 94.3% empty |
| Remove zero-guest rows | 180 | Not a bookable stay |
| Remove `adr` < 0 or ≥ 5000 | 2 | The two demonstrable errors only |
| Remove exact duplicate rows | 34,239 | See below |

119,390 → **84,969 bookings, 25 features**. No percentile-based outlier deletion: extreme is not invalid.

**Duplicates.** No booking ID exists, so a copy cannot be told apart from a genuine second booking.
Removing duplicates moved the cancellation rate from 37.1% to 27.7%. Keeping them scores **PR-AUC 0.9225
against 0.7660** (`ablation.csv` arm 7) — the model memorises copies it has already seen, which is not
skill. Limitation: some identical rows may be real separate bookings.

## 5. Feature engineering: measured, not assumed

Built: `total_nights` (weekday + weekend nights), `total_guests` (adults + children + babies),
`is_family` (children > 0 or babies > 0). Candidates from published Kaggle notebooks, scored on the
same folds (`scripts/ablation.py`), baseline **PR-AUC 0.7660 ± 0.0071**:

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

## 6. Preprocessing and split (notebook 02)

**Split.** 80/20 stratified train/test split, seed 42: train 67,975 rows, test 16,994 rows, cancellation
rate 27.7% in both. `test.csv` is opened only in notebook 08 (final test) and notebook 10 (ensemble check).
**Encoding.** `OneHotEncoder(min_frequency=100, handle_unknown='infrequent_if_exist')`. 142 countries and
277 agents have fewer than 100 bookings (2.7% and 5.0% of rows) and share one "infrequent" column.
**Scaling.** `StandardScaler` for Logistic Regression and the neural network only (trees do not need it).
**Nothing is fitted outside a fold** — encoder and scaler live inside each model's `Pipeline`.

**Class balance.** 27.7% positive. `class_weight='balanced'` / `scale_pos_weight` moved PR-AUC by
**−0.0100 to −0.0008, never positive** (notebook 09, `weight_sweep.csv`, `weighted_comparison.csv`,
`weighted_models.csv`). No SMOTE: the features are mostly categorical, so interpolation invents invalid
bookings. Imbalance is handled by choosing the decision threshold instead.

**Feature ranking.** Permutation importance, training split only, five folds
(`scripts/feature_importance_check.py` → `feature_importance.csv`). Top: `country` 0.173,
`lead_time` 0.137, `agent` 0.128, `total_of_special_requests` 0.107. Automatic feature selection was not
used: it promotes `required_car_parking_spaces`, a post-booking leak.

## 7. Models and results (notebooks 03–08)

Each model notebook tunes its settings with cross-validation on the training set only (5 outer folds ×
3 inner folds), picks a decision threshold from out-of-fold predictions, and saves the trained model to
`models/`.

| Model | Notebook | CV PR-AUC | Saved model |
| --- | --- | ---: | --- |
| XGBoost | 06 | **0.7612 ± 0.0090** | `models/xgboost.joblib` (3 MB) |
| Random Forest | 05 | 0.7548 ± 0.0079 | `models/random_forest.joblib` (48 MB) |
| Neural Network | 07 | 0.7426 ± 0.0111 | `models/neural_network.joblib` (0.6 MB) |
| Decision Tree | 04 | 0.7079 ± 0.0102 | `models/decision_tree.joblib` |
| Logistic Regression | 03 | 0.6470 ± 0.0095 | `models/logistic_regression.joblib` |
| Guessing (always the cancel rate) | 08 | 0.2775 | — |

**Final model: XGBoost** (600 trees, depth 10, learning rate 0.05, threshold 0.3352). Test set, notebook 08:
PR-AUC 0.7698, ROC-AUC 0.8924, F1 0.7019, recall 0.7692, precision 0.6454, accuracy 0.8187;
TN 10,286 · FP 1,993 · FN 1,088 · TP 3,627. It catches 77% of cancellations.

XGBoost leads Random Forest by 0.0064, which is inside the 0.0090 fold-to-fold spread, so the two are
close. XGBoost is preferred because it is about 12× faster to train and its file is 16× smaller.

## 8. Ensemble (notebook 10)

Soft voting: average the probabilities of Random Forest, XGBoost and the Neural Network.

| | XGBoost alone | Ensemble |
| --- | ---: | ---: |
| PR-AUC, 5 folds (training set) | 0.7660 | 0.7696 |
| PR-AUC, test set | 0.7698 | 0.7694 |
| F1, test set | 0.7019 | 0.7019 |
| Model file size | 3 MB | 52 MB |

The ensemble is ahead in all 5 folds, but only by +0.0037, which is under XGBoost's 0.0080 fold-to-fold
spread, and there is no gain on the test set. The three models' predictions are 92–94% correlated, so
averaging has little to correct. **XGBoost stays the final model.** The ensemble is saved as
`models/ensemble.joblib` for the record.

## 9. Why the ceiling (~0.76 PR-AUC) is not a modelling problem

- The three strongest models are within about one fold spread of each other.
- Twelve feature candidates together added +0.0026, below the noise.
- Class weighting never helped.
- Averaging three models (ensemble) added +0.0037 on training folds and nothing on test.
- The only large gains were leakage: post-booking columns (+0.0407) and duplicates (+0.155).

The limit is the information available when the booking is made, not the choice of model.

**Rule used for every comparison:** a change replaces the current best only if its gain is bigger than
the best model's fold-to-fold spread (the normal "noise" between folds), and every input is known when
the booking is made.

## 10. Limitations to state in the report

1. **Test set looked at more than once.** Earlier versions of the notebooks evaluated a test set of the
   same size, and notebook 10 reads the test set a second time (after the decision was made on training
   data), so the test score is not a perfectly untouched estimate.
2. **93 test bookings have an identical profile in training.** The random split cannot keep look-alike
   bookings on one side.
3. **Threshold is slightly optimistic.** It is picked on out-of-fold predictions from settings chosen
   on all training rows.
4. **No time-based validation.** Arrival date is not the booking-creation date.
5. **Scope.** Two Portuguese hotels, 2015–2017, before 2020.
6. **F1-best is not cost-best.** The threshold maximises F1; a hotel would set it from the real cost of
   contacting a guest against the value of a saved booking.

## 11. Contributions and AI use

The initial plan assigned M1 the shared foundation and Logistic Regression, M2 Decision Tree, M3 Random
Forest and M4 XGBoost. **In practice the project lead built the entire shared foundation alone** after
the schedule was compressed from seven weeks to three on 2026-09-03. The neural network and the ensemble
are additions beyond the four planned models. Member names and per-model ownership must be confirmed
before submission.

AI assistance was used to draft code and explanations, review the foundation, and run the ablation and
importance scripts. Reasoning and decisions were reviewed and accepted by the project lead.

Kaggle notebooks consulted for structure:

- [Hotel Booking Cancellation Prediction / EDA + 6 ML — Dalileh Oladzadeh](https://www.kaggle.com/code/dalileholadzadeh/hotel-booking-cancellation-prediction-eda-6-ml)
- [Hotel Cancellation Prediction using ANN — Aamir](https://www.kaggle.com/code/aamir5659/hotel-cancellation-prediction-using-ann)
- [HotelBookingCancellationAnalysis-V1 — Kushal](https://www.kaggle.com/code/kushal1147/hotelbookingcancellationanalysis-v1)
- [Hotel Booking — Hazem Alanany](https://www.kaggle.com/code/hazemalanany/hotel-booking)
- [Hotel Booking Cancellation Analysis / Python — Ahmed Baqa](https://www.kaggle.com/code/ahmedbaqa/hotel-booking-cancellation-analysis-python)

Their engineered features were measured (§5) rather than adopted on reputation (+0.0026 combined). Their
97–99% accuracy figures come from keeping outcome/post-booking columns, duplicates, or both.
