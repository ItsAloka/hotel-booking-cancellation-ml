# IT3091 ML — finishing plan (28 September 2026)

**Group 2026-AI-46 · Guided Data Track, code 6 (Hotel Booking Demand) · primary lens: cancellation risk.**

This is the working to-do list for a fresh session. History, numbers and past decisions are in
[`reports/project_record.md`](reports/project_record.md). Read it first. Also read
[`reports/decision_log.md`](reports/decision_log.md) and
[`reports/training_foundation.md`](reports/training_foundation.md).

- **Today (28 Sept):** tasks T1–T9. That covers the models, evaluation and supporting documents.
- **Tomorrow (29 Sept):** final report, 3-minute video, personal learning reports (section C).
- **Check the real deadline.** The old plan assumed 24 September, which has passed.

---

## A. Where we are (start state)

| Item | State |
| --- | --- |
| Data contract v2 | `data/processed/v2/` has `train.csv` (67,974), `test.csv` (16,995), `cv_folds.csv` and `manifest.json` with SHA-256 hashes. Split is group-aware: 0 shared profiles |
| Models | Dummy, LR, DT, RF, XGBoost, NN: nested CV (5 outer × 3 inner) in `notebooks/v2/04`–`09`; compared in `v2/10` |
| Reference model | **XGBoost** AP **0.7644 ± 0.0080** (600 rounds, depth 10, lr 0.05). RF 0.7552 ± 0.0082 |
| Out-of-fold probabilities | `reports/results/v2/oof/<model>.csv`, columns `row_id, outer_fold, is_canceled, proba`. `row_id` = 0-based row of `train.csv` |
| `test.csv` | **Never opened by any model.** It stays closed until T5 |
| Git | Nothing committed since 21 Sept (v2, scripts, logs, results all untracked) |

Rubric coverage today is about 63/100. Missing: ensemble, calibration, cost threshold, subgroups,
test pass, recommendation, diagram, canvas, data dictionary, AI-use declaration, reproducibility proof.

## B. Ground rules (apply to every task)

1. **Test set:** `data/processed/v2/test.csv` is read in **T5 only**, **once**, after the frozen
   decision is committed. No other notebook or script may read it.
2. **Stop/go rule 9.1** (`project_record.md` §9.1): a change replaces the reference only if its mean
   **paired** per-fold gain is greater than **0.0080** (XGBoost outer-fold SD, ddof=1) **and** it is
   positive in **≥ 4 of 5** outer folds. For cost, the gain in expected saving must be larger than the
   fold SD of saving. A failed check is still a result: record it with its number.
3. **Nothing is chosen on the fold it is scored on.** Every weight, calibrator and threshold is fitted
   on the other 4 outer folds' OOF rows and then applied to fold k ("cross-fitting").
4. **Every notebook starts by checking the manifest hashes** of `train.csv` and `cv_folds.csv`, the
   same way `v2/04`–`v2/10` do.
5. **Notebook style:** write it like a Kaggle kernel. Load the data first, one step per cell, and put
   a short markdown reading after every table or chart. No big config block at the top. Seed 42 everywhere.
6. **Do not reopen** feature engineering, SMOTE or undersampling, class weighting, or wider grids. The
   evidence is in `project_record.md` §5, §6 and §9.
7. **Logs:** every decision goes into `reports/decision_log.md` as a row: decision | alternatives |
   evidence/reason | limitation. Add a dated section per task.
8. Commit after each task (`git add` + commit, with the message naming the task).

---

## C. Today's tasks, in order

### T1. Commit the current state (10 min)

- `git status`: check that `data/processed/`, `Docs/` and `.devteam/` stay ignored (they are, in `.gitignore`).
- Stage: `notebooks/` (01, v1/, v2/), `reports/` (all logs, `results/`, `results/v2/`,
  `project_record.md`), `scripts/`, `plan.md`, `requirements.txt`, `.gitignore`, `data/README.md`,
  `data/raw/SOURCE.md`. The deletions of the old `notebooks/01_eda_and_cleaning.ipynb`,
  `02_pipeline_and_split.ipynb` and `reports/feature_roles.csv` are intended.
- `reports/data_dictionary.csv` shows as deleted. **Do not commit that deletion.** Restore it with
  `git checkout -- reports/data_dictionary.csv`; it is rebuilt in T6.
- Commit: `v2 foundation: group-aware split, nested-CV retrain of six models, project record`.
- **Done when:** `git status` is clean apart from ignored files.

### T2. Ensemble — `notebooks/v2/11_ensemble.ipynb` (≈1 h)

Uses only the saved OOF files; no model is refit.

1. **Diagnose.** Load the OOF files for XGB, RF, NN and DT and join them on `row_id`. Report the
   correlation of the probabilities, the correlation of the residuals (`is_canceled − proba`) for
   XGB vs RF and XGB vs NN, and the disagreement rate at 0.5. Residual correlation above about 0.9
   means little complementary signal. Say so, but still run the three options below, because the
   plan requires the ensemble to be reported.
2. **Options** (per outer fold k, score on fold k):
   - **E1 simple average:** `(p_xgb + p_rf) / 2`.
   - **E2 weighted average:** `w·p_xgb + (1−w)·p_rf`, with `w ∈ {0.3, 0.4, 0.5, 0.6, 0.7}` chosen by
     pooled AP on the **other 4 folds**. Record the chosen `w` per fold.
   - **E3 stacking:** `LogisticRegression` on `logit(p)` of XGB, RF, NN and DT, fitted on the other 4
     folds' OOF rows and scored on fold k.
     **Exploratory only, not adoptable.** The OOF values of the other 4 folds came from base models
     that were trained on fold k's rows, so fold k's labels reach the meta-learner indirectly. This
     breaks rule B3. Removing the leak would mean refitting the base models without fold k, which T2
     does not do. Report E3's numbers, but it cannot replace XGBoost.
3. **Compare with XGBoost** using rule 9.1: per-fold AP, mean ± SD, paired gain, folds better. Also
   report precision, recall and F1 at 0.5.
4. **Outputs:** `reports/results/v2/ensemble.csv` (one row per option × fold plus mean/sd rows) and
   `reports/results/v2/oof/ensemble_<name>.csv` (same columns as the other OOF files) for each option.
5. **Decision:** adopt E1 or E2 as the final model only if it passes rule 9.1 (E3 cannot be adopted). Expected outcome from
   the evidence: roughly +0.002–0.005, which fails, so **XGBoost stays**. Record this as further
   evidence of the information ceiling.
6. Decision-log section "T2 ensemble"; commit.
- **Done when:** the table exists, a final model is named, and the log row is written.

### T3. Calibration and cost-based threshold — `notebooks/v2/12_calibration_threshold.ipynb` (≈1.5 h)

Input: the OOF of the final model from T2 (XGBoost unless T2 said otherwise), joined to `train.csv`
by `row_id` to get `adr`, `total_nights`, `hotel` and so on.

**Part 1: calibration**

1. Brier score (per fold and pooled) and a reliability diagram (10 quantile bins) for raw XGBoost.
2. Cross-fitted **isotonic** and **Platt (sigmoid)** calibrators: fit on the other 4 folds' OOF and
   apply to fold k. Report Brier per fold for raw, isotonic and Platt, plus reliability diagrams.
   AP barely changes, because calibration is monotone. Check that too.
3. Adopt a calibrator only if the paired per-fold **reduction** `d_k = Brier_raw,k − Brier_cal,k`
   has a mean greater than the fold SD of raw Brier (ddof=1) **and** `d_k > 0` in ≥ 4/5 folds
   (rule 9.1 applied to Brier; lower Brier is better). Otherwise keep raw probabilities and state how
   well calibrated they are.

**Part 2: cost scenario.** The values are assumptions, and the notebook and decision log must say so.

- Room value at stake: `V = adr × total_nights` (euros). Day-use rows have V = 0.
- Flagging a booking triggers a contact action (reminder, deposit or confirmation request) that costs
  **c** per booking.
- If a flagged booking really cancels, the hotel recovers a fraction **r** of V (earlier resale or
  controlled overbooking).
- Net saving of flagging booking *i* = `r·V_i·y_i − c`. Not flagging = 0.
- Base case: **c = €5, r = 0.25**. Sensitivity grid: c ∈ {2, 5, 10, 20} × r ∈ {0.10, 0.25, 0.50}.

**Part 3: decision rules** (cross-fitted: choose on 4 folds, evaluate on fold k)

- **R1 global threshold:** flag if `p ≥ t`, with t on a 0.05–0.95 grid chosen to maximise saving.
- **R2 value-aware rule:** flag if `p·r·V ≥ c`. This needs calibrated probabilities and has no
  parameter to tune.
- **R3 F1-optimal threshold:** reported for comparison only.
- For each rule and fold, report total saving (€), saving per 1,000 bookings, % of bookings flagged,
  precision, recall and F1.
- **Stop/go, R2 vs R1:** R1 is the reference. Let `g_k = S_R2,k − S_R1,k`, where S is the saving per
  1,000 bookings on fold k, so fold size does not matter. R2 replaces R1 only if mean `g_k` > the fold
  SD of `S_R1` (ddof=1) **and** `g_k > 0` in ≥ 4/5 folds. Otherwise R1 stays. In the sensitivity
  table, use the same test against "flag nothing" (saving 0).

Then:

4. **Sensitivity table:** the best rule's saving and threshold for every (c, r) cell. Check whether
   the recommendation still holds across the grid, and mark the cells where flagging nothing is best.
5. **Freeze:** calibrator (or none), rule, threshold, and base-case c and r. Write
   `reports/results/v2/threshold.json` and `calibration.csv`, `cost_sensitivity.csv`, `decision_rules.csv`.
6. Decision-log section "T3"; commit.
- **Done when:** one rule and its parameters are frozen, with sensitivity evidence.

### T4. Subgroup check — `notebooks/v2/13_subgroups.ipynb` (≈30 min)

Input: the final OOF probabilities (calibrated if adopted) and the frozen T3 rule, joined to `train.csv`.

1. For each group of `hotel`, `distribution_channel`, `market_segment`, `customer_type`,
   `deposit_type`, `is_repeated_guest`, report: support n, cancel rate, AP, precision, recall, % flagged,
   and saving per booking.
2. Flag any group whose recall is more than **0.10 below** the overall recall. Groups with **n < 500**
   are "indicative only".
3. `country`: the top 10 by volume only, with the caveat that it describes booking origin, not people.
   Make no nationality claims. `country` is a model input, which is itself a fairness point to raise
   in limitations.
4. Output `reports/results/v2/subgroups.csv`. Add a decision-log section, commit.
- **Done when:** the table exists and the weak groups are named in the log.

### T5. Freeze, final fit, one test pass — `notebooks/v2/14_final_model_test.ipynb` (≈1 h)

**Before any test read:**

1. Write `reports/results/v2/frozen_model.json` containing:
   - final model and hyperparameters (XGBoost 600 / 10 / 0.05, `random_state=42`, the same pipeline
     as `v2/08`: `OneHotEncoder(min_frequency=100, handle_unknown='infrequent_if_exist')`, no scaling)
   - calibrator
   - decision rule and threshold
   - c and r
   - the `train.csv` and `test.csv` hashes from the manifest
2. **Commit `frozen_model.json` on its own first.** The commit timestamp proves the decision came
   before the test set was opened.

**Then, in the notebook:**

3. Refit the final pipeline on all 67,974 training rows.
   - If a calibrator was adopted, fit it on the pooled OOF probabilities from T3. State the small
     mismatch: the OOF came from per-fold models, and the final model is a full refit.
   - Time the refit.
4. Check the hash of `test.csv`, then load it **once** and score it. Report:
   - AP, ROC-AUC and Brier
   - precision, recall, F1, accuracy and confusion matrix at the frozen rule
   - saving under the base case and under the sensitivity grid
   - the subgroup table from T4 repeated on test (hotel and channel at least)
   - PR curve and reliability diagram
5. Compare test AP with the nested-CV AP (0.7644 ± 0.0080). If test AP falls inside about ±2 SD,
   the CV estimate held. Say whether it did either way, and do not retune.
6. Save `reports/results/v2/final_test.csv` and `final_test_subgroups.csv`, plus figures under
   `reports/figures/` (PR curve, reliability diagram, confusion matrix, cost sensitivity heatmap,
   leaderboard bar chart with SD whiskers). The report and video will use these figures.
7. Optional: save the model to `models/final_xgb.joblib` and add `models/` to `.gitignore`.
8. Decision-log section "T5 frozen decision and single test pass"; commit.
- **Done when:** test metrics are saved and the log says the test set was opened exactly once.

### T6. Data dictionary — `reports/data_dictionary.csv` (+ short `.md`) (≈30 min)

- Start from the restored v1 CSV (columns: `column, dtype, non_null, missing, missing_%, n_unique,
  example, meaning, role`).
- Update it for the final contract:
  - all 32 raw columns
  - `role` ∈ {predictor, target, excluded-leakage, excluded-post-booking, excluded-other, engineered}
  - a `reason` column for every exclusion
  - the 3 engineered features (`total_nights`, `total_guests`, `is_family`) with formulas
- Row meaning: **one row = one hotel booking (unique predictor profile after deduplication)**.
- Add the dataset link, source DOI and the raw-file SHA-256 at the top of `reports/data_dictionary.md`.

### T7. Framing canvas and workflow diagram (≈45 min)

**`reports/problem_framing_canvas.md`**, a one-page table with:
- Stakeholder: hotel revenue manager
- Decision need: whom to contact / how much to overbook
- Primary lens: cancellation risk
- Secondary lens: none, and why (segmentation would not strengthen the contact decision; the
  subgroup check covers operational differences)
- Unit of analysis: a booking at creation time
- Task: binary classification, calibrated probability + flag
- Output
- Success metric: AP for ranking, expected € saving for the decision
- Constraints: booking-time inputs only, no leakage, interpretability
- Data, risks and ethical notes

**`reports/workflow_diagram.md`**, a Mermaid flowchart:
- business problem → data source and fingerprint → EDA → cleaning (duplicates, errors) → leakage
  exclusions → group-aware split (test locked) → nested CV of 6 models → reference selection (rule 9.1)
  → ensemble test → calibration + cost threshold → subgroup check → freeze → single test pass →
  recommendation
- Mark the decision points and link each to its decision-log section.
- Export a PNG to `reports/figures/workflow.png` (mermaid.live or `mmdc`) for the report and video.

### T8. Fix the logs, and write the recommendation, limitations and AI-use (≈1 h)

1. **Decision log.** The "17 September 2026" section contradicts the real contract. It says 118,565
   rows, 20 predictors, duplicates preserved, and ADR excluded; the real contract is 84,969 rows,
   25 predictors, duplicates removed, and ADR kept.
   - Mark that section as **superseded** (a draft that was not adopted).
   - Add a correct "17 September — v1 foundation" section summarised from `project_record.md` §2–§6,
     with the measured numbers.
2. **`reports/training_foundation.md`:** make sure it only describes files that exist.
3. **`reports/recommendation_and_limitations.md`**, filled with the T3–T5 numbers:
   - **Recommendation:** use the frozen rule to rank new bookings daily and contact the flagged ones.
     Give the expected saving per 1,000 bookings, the % contacted, and the precision (how many
     contacts are wasted).
   - Subgroup caveats from T4.
   - Retraining trigger: quarterly, or when the monthly cancel rate drifts by more than 5 points.
   - **Not for automatic cancellation or price discrimination.** A human stays in the loop.
   - **Limitations:**
     - retrospective data only (two Portuguese hotels, 2015–2017, pre-2020)
     - no temporal validation
     - the CSV is not a verified booking-time snapshot
     - the unique-profile population (duplicates)
     - the cost values are assumptions
     - `country` as an input raises a fairness question
     - the test population was seen during EDA (no labels were used for model choices)
   - Stakeholder value: what the manager gains, and what they must not over-read.
4. **`reports/ai_use_declaration.md`:** the tools used (Claude Code, Codex via DevTeam), what they did
   (drafted code and text, reviews, ran scripts), and what humans decided and checked. Say that all
   numbers come from the notebooks, not from AI output. Follow the course rules.

### T9. Clean reproducibility run (≈1 h, mostly waiting)

1. Create a fresh venv: `python -m venv .venv-check`, then `pip install -r requirements.txt`.
2. Execute in order with `jupyter nbconvert --to notebook --execute --inplace`:
   - `01_eda_cleaning`
   - `v2/02` → `v2/03` → `v2/04` … `v2/14`
   - RF (`v2/07`) takes about 30 min, so run it in the background.
3. Check:
   - the manifest hashes are identical
   - `git diff --stat reports/results/v2/` shows no metric changes beyond float noise
   - `final_test.csv` is identical
4. Record the result, Python version and machine in `reports/reproducibility.md`, including the
   dataset link and fingerprint.
5. Final commit for today: `Models, evaluation and supporting documents complete`.
- **Done when:** a clean run reproduces every saved number.

### Today's checklist

| # | Task | Output | Status |
| --- | --- | --- | --- |
| T1 | Commit state | clean git | ☑ |
| T2 | Ensemble | `v2/11`, `ensemble.csv` | ☐ |
| T3 | Calibration + cost threshold | `v2/12`, `threshold.json` | ☐ |
| T4 | Subgroups | `v2/13`, `subgroups.csv` | ☐ |
| T5 | Freeze + test pass | `frozen_model.json`, `v2/14`, `final_test.csv`, figures | ☐ |
| T6 | Data dictionary | `data_dictionary.csv/.md` | ☐ |
| T7 | Canvas + workflow | `problem_framing_canvas.md`, `workflow_diagram.md`, `workflow.png` | ☐ |
| T8 | Logs, recommendation, AI use | `decision_log.md`, `recommendation_and_limitations.md`, `ai_use_declaration.md` | ☐ |
| T9 | Clean rerun | `reproducibility.md` | ☐ |

**Parallel work if two agents run:** T6 and T7 do not depend on T2–T5 and can run alongside them.
T8's recommendation needs T5. T9 runs last.

---

## D. Tomorrow (29 September) — not today

1. **Final report**, structured by the rubric sections:
   - framing (5)
   - workflow and decision log (10)
   - EDA and data quality (10)
   - preprocessing (15)
   - model comparison (20)
   - evaluation (20)
   - recommendation (10)
   - reproducibility and AI use (10)
   Every number cites its CSV.
2. **3-minute YouTube demo:** script, then slides/figures from `reports/figures/`, then recording.
3. **One A4 Personal Learning Journey report per member.** Each member writes their own; prepare a
   template only.
4. Submission package: report, notebooks and code, dataset link and fingerprint, data dictionary,
   decision logs, model comparison.
5. Confirm member names and contributions (`project_record.md` §11).
