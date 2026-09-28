# Decision log

One row per decision: what we chose, what else we considered, why, and what it does not solve.
Numbers come from the notebooks and `reports/results/`; see [project_record.md](project_record.md) for detail.

## Data and cleaning (notebook 01)

| Decision | Alternatives considered | Evidence / reason | Limitation |
| --- | --- | --- | --- |
| Remove exact duplicate rows (34,239) | Keep them | Keeping them gives PR-AUC 0.9225 against 0.7660 because the model memorises copies | Some copies may be real separate bookings; there is no booking ID |
| Remove zero-guest rows (180) and the 2 impossible prices | Delete outliers by percentile | Only rows that cannot be a real stay are removed; extreme is not the same as wrong | Other odd values stay in |
| Missing `country` → `Unknown`, `agent` → 0 (as text), `children` → 0 | Fill with the most common value | Missing agent means "no agent"; an unknown country should not be guessed | `children` = 0 is an assumption for 4 rows |
| Drop leak columns (`reservation_status*`, parking, assigned room, booking changes, waiting list) | Keep them for a higher score | They are only known after the booking is made; together worth +0.0407 PR-AUC, which would be leakage | The CSV cannot prove every kept column was fixed at booking time |
| Drop `company`, `arrival_date_year`, `arrival_date_week_number` | Keep them | 94.3% empty / partial years / 0.995 correlated with month | — |

## Features and preprocessing (notebook 02, scripts)

| Decision | Alternatives considered | Evidence / reason | Limitation |
| --- | --- | --- | --- |
| Add `total_nights`, `total_guests`, `is_family` | Raw columns only | Easy to read; no measured harm | `is_family` adds almost nothing (0.0001) |
| Do not add the Kaggle notebooks' extra features | Add them all | All together +0.0026, below the 0.0071 noise (`ablation.csv`) | Tested with one model only |
| 80/20 stratified split, seed 42 | Other ratios | Keeps the 27.7% cancel rate on both sides | 93 test bookings look identical to a training booking |
| One-hot encoding, rare categories (<100 bookings) share one column | Frequency encoding | Frequency encoding scored −0.0010 | Threshold 100 was chosen, not tuned |
| Scale numbers only for Logistic Regression and the Neural Network | Scale for all | Tree models are not affected by scale | — |
| No class weighting, no SMOTE (notebook 09) | `class_weight='balanced'`, SMOTE | Weighting lowered PR-AUC in every model (−0.0100 to −0.0008); SMOTE invents impossible bookings from categorical data | Imbalance handled by the threshold instead |

## Models (notebooks 03–08)

| Decision | Alternatives considered | Evidence / reason | Limitation |
| --- | --- | --- | --- |
| Rank models on PR-AUC | Accuracy | Always answering "not cancelled" already scores 72% accuracy; PR-AUC focuses on the cancelled class | Harder to explain than accuracy |
| Nested cross-validation (5 outer × 3 inner folds) on the training set | Tune and score on the same folds | The score used to compare models is not the one used to tune them, so it is not flattered | Slower (Random Forest ~25 min) |
| Threshold chosen by best F1 on out-of-fold training predictions | Default 0.5 | Raises XGBoost F1 from 0.673 to 0.698 | F1-best is not cost-best |
| **Final model: XGBoost** (0.7612 ± 0.0090) | Random Forest (0.7548) | Best score; RF is within the noise but 12× slower and 16× bigger | The margin over RF is not significant |
| Test set used once for the final model (notebook 08): PR-AUC 0.7698, accuracy 81.9%, recall 76.9% | — | Test score lands inside the CV spread, so the CV estimate was honest | — |
| Save every trained model to `models/` with joblib | Retrain whenever needed | A saved model can score new bookings straight away | Random Forest file is 48 MB |

## Ensemble (notebook 10, 28 September 2026)

| Decision | Alternatives considered | Evidence / reason | Limitation |
| --- | --- | --- | --- |
| Try soft voting of RF + XGBoost + NN | Add LR and DT too | LR and DT are clearly weaker and would pull the average down | Only one ensemble type tried |
| **Keep XGBoost, do not adopt the ensemble** | Adopt the ensemble | +0.0037 PR-AUC on training folds (5/5 folds better) is under the 0.0080 noise; test PR-AUC 0.7694 vs 0.7698, same F1; 52 MB vs 3 MB | Test set read a second time, after the decision |
