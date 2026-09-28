# Decision log

## 17 September 2026 — revised EDA, cleaning and preprocessing

| Decision | Alternatives considered | Evidence / reason | Limitation |
| --- | --- | --- | --- |
| Historical snapshot claim | Initial-booking / day-before-arrival deployment claim | Source timing cannot verify original values or all outcome ordering | No live deployment claim |
| Exclude ADR, deposits, requests in default | Retain conditional snapshot inputs | Conservative response to unverified timing | Other retained fields also may be amended |
| Overnight population, >=1 guest | Keep administrative/day-use rows | Align scope with overnight room planning | 825 excluded records remain auditable |
| Preserve duplicate records and group profiles | Deduplicate every matching row | No booking ID; avoid assuming records are redundant | True multiplicity unresolved; group keys not guest IDs |
| Company presence flag | Drop company altogether / retain every ID | Source-backed business relationship, compact representation | Essentially no incremental fixed-tree AP |
| Totals replacing components | Raw components / totals alongside all components | Interpretable units without exact sum redundancy | Raw components slightly better in fixed-tree checks; not performance optimisation |
| Keep family/child-missing indicators as candidates | Add on univariate association alone | Tiny mixed / zero AP changes on training folds | Other estimators could behave differently |
| Shared one-hot builder | Whole-frame frequency encoding | Fold-local fitted transforms and readable categories | Threshold 100 is pre-specified, not tuned optimum |
| Five outer / three inner grouped folds | Ordinary stratification | No matching predictor profile across evaluation boundaries | Historical comparison only |
| Versioned v2 outputs | Overwrite old processed/results files | Preserve provenance and make retraining needs explicit | Historical model notebooks require migration |

Recorded result: 61 checks passed; 118,565 cleaned rows, 20 predictors. No test classifier score was used. See plan.md and the feature log for full-precision supporting CSVs and remaining work.

## 21 September 2026 — v2 step 1: group-aware split

Notebook: `notebooks/v2/02_preprocessing_v2.ipynb`. Output: `data/processed/v2/` (`train.csv`, `test.csv`, `manifest.json`).

| Decision | Alternatives considered | Real-world reason | Limitation |
| --- | --- | --- | --- |
| v1 notebooks 02–09 moved to `notebooks/v1/`; 01 stays shared in `notebooks/` | Edit v1 in place | Notebook 02 changes in v2; keeping v1 intact lets anyone rerun the numbers already reported. Only relative paths were edited (`../` → `../../`) | v1 results remain historical, not comparable with v2 |
| Keep the v1 population (duplicates removed, 84,969 rows) and the same 25 predictors | Restore duplicates; add features | Changing one thing at a time means any score change can be blamed on the split alone | Still a "unique profiles" population, not "all bookings" |
| Group key = exact match on every predictor column (target excluded, `agent` as text) | Key on a subset of columns | Two bookings that look identical to the model are, for the model, the same booking; the target is left out so the grouping cannot see the answer | 313 profiles are shared by 626 rows; near-duplicates are not grouped |
| `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, fold 0 = test | Random `train_test_split`; `GroupShuffleSplit` | Like a hotel predicting next season's bookings, the test must hold bookings the model has never seen. 5 folds keeps v1's 80/20 size; stratifying keeps the 27.75% cancel rate on both sides; seed 42 as everywhere else | Fold 0 is fixed in advance, not chosen by score |
| Result: train 67,974 / test 16,995 rows, cancel rate 0.2775 / 0.2775; profile overlap 93 (v1, recounted) → 0 (v2) | — | Removes the memorisation route that inflated v1 test scores | — |
| Manifest with row counts, cancel rates, seed, group key and SHA-256 hashes | No manifest | A number the hotel acts on must be reproducible; the hashes prove the files were not changed. Rerun reproduced identical hashes | — |
| Later v2 CV must group by the same profile inside training | Plain stratified CV | Same leak otherwise reappears between fitting and validation folds | To be done in the next v2 step |

No test-set information was used for any decision; the test file was only written and hashed.

## 21 September 2026 — v2 step 2: nested cross-validation folds

Notebook: `notebooks/v2/03_cv_folds_v2.ipynb`. Output: `data/processed/v2/cv_folds.csv`; hash and settings added to `manifest.json`.

| Decision | Alternatives considered | Real-world reason | Limitation |
| --- | --- | --- | --- |
| Group the folds by the same profile key as the split | Plain `StratifiedKFold` inside training | If a profile sits in both the fitting and validation part, the validation score rewards memory, and a model tuned on that score is picked for the wrong reason | Near-duplicate profiles are still not grouped |
| 5 outer × 3 inner `StratifiedGroupKFold`, shuffle, seed 42 | Single 5-fold CV for both tuning and comparison | Tuning and scoring on the same folds flatters the tuned model. The inner folds choose settings; the outer fold, never seen during tuning, scores them, like a hotel judging a model on bookings it was not tuned on | 5 × 3 fits per setting is slower |
| Save the folds to a file with a hash | Recreate folds in each model notebook | Every model is compared on exactly the same rows, and anyone can check the file was not changed | Notebook 02 rewrites the manifest, so 03 must run after it |
| One row per training row: `row_id`, `outer_fold`, `inner_fold_0`–`inner_fold_4` | One `inner_fold` column | A row is fitted in 4 of the 5 outer folds, so it needs an inner fold for each of them | — |
| Result: 0 shared profiles in all 5 outer and 15 inner pairs; cancellation rate 0.2774–0.2775 everywhere; rerun of 01 → 02 → 03 reproduced every hash | — | — | Fresh-environment install not tested |

Fold creation read only `train.csv` (plus `manifest.json`, which it extends); neither the cleaned source table nor `test.csv` was opened.

## 21 September 2026 — v2 step 3: retrain every model on the frozen contract (plan §9.3)

Notebooks: `notebooks/v2/04_dummy_baseline` … `09_neural_network`, `10_model_selection`. Outputs: `reports/results/v2/<model>.csv` (one row per outer fold, then mean and SD rows), `reports/results/v2/oof/<model>.csv` (67,974 rows, `row_id`, `outer_fold`, `is_canceled`, `proba`), `leaderboard.csv`, `reference_selection.csv`.

| Decision | Alternatives considered | Real-world reason | Limitation |
| --- | --- | --- | --- |
| Nested CV strictly from `cv_folds.csv`: tune on `inner_fold_k` inside outer fold k's fitting rows, score the untouched fold k; each notebook checks the manifest SHA-256 of `train.csv` and `cv_folds.csv` first | Re-draw folds with `StratifiedKFold` in each notebook (v1) | Every model is judged on exactly the same bookings, and no profile crosses a tuning or scoring boundary | 5 × 3 fits per setting; RF took ~30 min |
| v1 grids and search types unchanged (LR/DT full grid; RF 6, XGB 10, NN 3 sampled with seed 42) | Widen grids | Changing one thing at a time: the new split, not new tuning, explains any score change | LR picked C = 10, the grid edge, in 3 of 5 folds; XGB picked the largest `n_estimators` (600) everywhere. Neither grid was widened (rule 9.1: no tuning chase without evidence) |
| Out-of-fold probabilities = each outer fold scored by the model tuned without it | Refit one final model and `cross_val_predict` it (v1) | The v1 route reused settings chosen on all rows, so its OOF scores had seen the answer; stages 9.4–9.6 need honest OOF probabilities | — |
| Precision / recall / F1 on the cancelled class at a 0.5 threshold | Tuned F1 threshold | The threshold is a business choice made on cost in §9.4, not here | Dummy has 0 precision/recall at 0.5 by construction |
| SD reported with ddof = 1 (sample SD over 5 folds) | ddof = 0 (v1 `np.std`) | Rule 9.1's noise bar should not understate the spread from only 5 folds | Not directly comparable with v1 SDs |
| Fit time = tuning + refit (`search_seconds`, parallel over candidates, machine-dependent); `refit_seconds` = one refit; `predict_seconds` = scoring one outer fold | Single timing | The refit time is what a hotel would pay to retrain | Timed on one 20-core laptop |
| **Reference model: XGBoost**, AP 0.7644 ± 0.0080 | Random Forest 0.7552 ± 0.0082 (tie-break to simpler) | Rule 9.1 walked LR → DT → RF → XGB → NN. XGB beat RF by +0.0092 against a 0.0082 bar, 5/5 folds; about 30× faster to refit and higher F1 at 0.5 (0.674 vs 0.631). NN was −0.0207, 0/5 | The margin over RF is barely above the bar; §9.6 should treat RF as the natural ensemble partner, not as beaten decisively |

Other results (AP mean ± SD): Dummy 0.2775, Logistic Regression 0.6410 ± 0.0091, Decision Tree 0.7067 ± 0.0069, Neural Network 0.7437 ± 0.0101. `test.csv` was not opened by any notebook.

## 28 September 2026 — T2 ensemble

Notebook: `notebooks/v2/11_ensemble.ipynb`. Outputs: `reports/results/v2/ensemble.csv` (option × outer fold, then mean and SD rows, ddof=1; XGBoost included as the paired reference) and `reports/results/v2/oof/ensemble_e1_average.csv`, `ensemble_e2_weighted.csv`, `ensemble_e3_stacking.csv`. Built only from the saved OOF files of XGBoost, Random Forest, Neural Network and Decision Tree; no model was refit.

| Decision | Alternatives considered | Evidence / reason | Limitation |
| --- | --- | --- | --- |
| **XGBoost stays the final model** | E1 average of XGB and RF; E2 weighted average; E3 stacking | Rule 9.1 bar = 0.0080 (XGBoost fold SD) and ≥ 4/5 folds. E1: AP 0.7693 ± 0.0081, paired gain +0.0049, 5/5 folds → fails. E2: 0.7692 ± 0.0080, +0.0048, 5/5 → fails. A blend would also add RF's ~191 s refit to XGBoost's ~6 s and be harder to explain | The ensembles are better in every fold; the gain is consistent in sign but about 60% of the noise bar, so it is not claimed |
| Diagnose before blending | Blend blindly | Residual correlation (`is_canceled − proba`) 0.94 for XGB–RF and XGB–NN; probability correlation 0.93 / 0.92; the models disagree on the 0.5 flag for 7.2% / 7.7% of bookings. Little complementary signal, so only a small gain was expected | Pearson correlation on all OOF rows pooled |
| E2 weight `w ∈ {0.3,…,0.7}` chosen by pooled AP on the other 4 folds (cross-fitted, rule B3) | Choose `w` on all folds | Chosen `w` = 0.5, 0.6, 0.6, 0.5, 0.5 for folds 0–4; the AP curve over `w` is flat (≤ 0.0002 between 0.5 and 0.6), so E2 ≈ E1 | Grid of five weights only |
| E3 stacking (logistic regression on logit p of XGB, RF, NN, DT, fitted on the other 4 folds) reported as **exploratory only, not adoptable** | Adopt it if it passes | The other folds' OOF values came from base models trained on fold k's rows, so fold k's labels reach the meta-model indirectly. Even with that help: +0.0062, 5/5 → fails. Coefficients ≈ 0.48 XGB, 0.46 RF, 0.15 NN, 0.02 DT | Removing the leak would need base models refitted without fold k; not done |
| Precision / recall / F1 at 0.5 reported, not used to decide | Decide on F1 | E1 trades recall for precision (0.748 / 0.598 vs XGBoost 0.727 / 0.628; F1 0.665 vs 0.674) because averaging with RF pulls probabilities towards the middle. The threshold is chosen on cost in T3 | Untuned threshold |

Further evidence of an information ceiling in the 25 booking-time predictors: four model families make largely the same mistakes. `test.csv` was not opened; only the label column of `train.csv` and `cv_folds.csv` were read, after their manifest hashes were checked.

## 28 September 2026 — T3 calibration and cost threshold

Notebook: `notebooks/v2/12_calibration_threshold.ipynb`. Outputs: `reports/results/v2/threshold.json` (the frozen decision), `calibration.csv` (Brier and AP per fold for raw, isotonic and Platt, then mean and SD rows, ddof=1), `decision_rules.csv` (R1/R2/R3 × outer fold at the base case, then mean and SD rows) and `cost_sensitivity.csv` (12 cost cells). Built only from `reports/results/v2/oof/xgboost.csv` joined to `train.csv` (`hotel`, `adr`, `total_nights`); no model was refit. Every calibrator and threshold was fitted on the other 4 outer folds and scored on fold k.

**The cost values are assumptions, not hotel data:** flagging costs c per booking; a flagged booking that cancels recovers r·V, with V = `adr × total_nights` (day-use V = 0); saving = r·V·y − c if flagged, 0 if not. Base case c = €5, r = 0.25. Grid c ∈ {2, 5, 10, 20} × r ∈ {0.10, 0.25, 0.50}.

| Decision | Alternatives considered | Evidence / reason | Limitation |
| --- | --- | --- | --- |
| **No calibrator: raw XGBoost probabilities are kept** | Isotonic; Platt (sigmoid on logit p) | Rule 9.1 applied to Brier, bar = 0.0020 (SD of raw Brier). Platt: mean reduction 0.00027, 5/5 folds → fails. Isotonic: 0.00017, 4/5 → fails, and it costs 0.0078 AP because its steps create ties. Raw is already well calibrated: pooled Brier 0.1176 (constant guess 0.2005), largest reliability-bin gap 0.03, mean p 0.2727 vs cancel rate 0.2775 | Mild S-shape remains: under-predicts by ~0.02 in the 0.04–0.2 range, over-confident by ~0.03 in the top bin |
| **Rule R1: flag if p ≥ 0.05** (t chosen on the pooled OOF; the cross-fitted t was 0.05 on all 5 folds) | R2 value-aware (flag if p·r·V ≥ c); R3 F1-optimal threshold | Base case, cross-fitted: R1 saves €29,601 ± 826 per 1,000 bookings, flags 63.5%, precision 0.43, recall 0.97, F1 0.59. R2: €29,853 ± 841; paired gain +€252, 5/5 folds, against an €826 bar → fails, so R1 stays. R3 (t ≈ 0.36): best F1 0.70, but €4,900 per 1,000 bookings worse than R1 in every fold | **t = 0.05 is the lowest value on the grid**, so the unconstrained optimum is at or below it. With V ≈ €400 on average, a caught cancellation recovers about 24× the contact cost, and contacting everyone already saves €28,300 per 1,000 bookings. The model adds only about €1,300 over that, by skipping the lowest-risk ~37% |
| F1 not used to set the threshold | Choose t by F1, as the untuned 0.5 cut-off in T2 did | A missed cancellation costs far more than a wasted contact under every cost cell tested, so F1's equal weighting is the wrong target | — |
| **Sensitivity: the recommendation (act on the model, do not flag nothing) holds in all 12 cells** | — | Flagging nothing is never best: every cell's rule saves money in 5/5 folds, far above its fold SD. R1 wins 9 of 12 cells. R2 wins where contact is dear and recovery low (c = €10, r = 0.10; c = €20, r = 0.10 and 0.25), gaining up to €1,300 per 1,000 bookings. R2 beats R1 in 5/5 folds in every cell but usually by less than the bar | The threshold itself is not robust: R1's pooled t ranges from 0.05 to 0.47 across the grid. It is driven by c and r more than by the model. The hotels' real contact cost and recovery rate are needed before t is used |

`test.csv` was not opened. T5 uses raw XGBoost probabilities and flags bookings with p ≥ 0.05; there is no calibrator to refit.
