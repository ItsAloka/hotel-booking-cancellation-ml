# IT3091 ML — plan

**Group 2026-AI-46 · Guided Data Track, code 6 (Hotel Booking Demand) · predicting booking cancellations.**

Numbers and decisions: [`reports/project_record.md`](reports/project_record.md) and
[`reports/decision_log.md`](reports/decision_log.md). Assignment brief: `Docs/ML Assignment.pdf`.

## Where we are (28 September 2026)

The ML work is **finished**: collect data → clean → train and compare models → pick one → save it.

| Step | Notebook | Output |
| --- | --- | --- |
| EDA and cleaning | `01_eda_cleaning` | `data/processed/hotel_bookings_clean.csv` (84,969 bookings) |
| Preprocessing and train/test split | `02_preprocessing` | `data/processed/train.csv`, `test.csv` |
| Logistic Regression | `03_logistic_regression` | `models/logistic_regression.joblib` |
| Decision Tree | `04_decision_tree` | `models/decision_tree.joblib` |
| Random Forest | `05_random_forest` | `models/random_forest.joblib` |
| XGBoost | `06_xgboost` | `models/xgboost.joblib` |
| Neural Network | `07_neural_network` | `models/neural_network.joblib` |
| Model selection + final test | `08_model_selection` | `reports/results/final_model.csv` |
| Class weighting check | `09_class_weighting` | `reports/results/weighted_*.csv` |
| Ensemble (soft voting) | `10_ensemble` | `models/ensemble.joblib` |

**Final model: XGBoost.** Test PR-AUC 0.770, ROC-AUC 0.892, accuracy 81.9%, catches 77% of cancellations.
Every honest approach lands at about 0.76–0.77 PR-AUC; that is the ceiling of the data, not a mistake.

What is left is the **written evidence** the rubric asks for, the video and the personal reports.

---

## Part 1 — finish the evidence documents (today)

Each item is required by the assignment brief (section 4, "Core evidence required from every group").
All files go in `reports/`.

| # | Task | File | Rubric criterion (marks) | Done? |
| --- | --- | --- | --- | --- |
| 1 | **Problem framing canvas** | `reports/problem_framing_canvas.md` | Business problem framing (5) | ☐ |
| 2 | **Workflow diagram** | `reports/workflow_diagram.png` (+ source) | Workflow diagram and decision log (10) | ☐ |
| 3a | **EDA insight log** | `reports/eda_insight_log.md` | Data understanding, EDA (10) | ☐ |
| 3b | **Preprocessing / feature log** | `reports/preprocessing_feature_log.md` | Preprocessing and feature engineering (15) | ☐ |
| 4 | **Recommendation + limitations** | `reports/recommendation.md` | Recommendation and stakeholder value (10) | ☐ |
| 5 | **AI-use declaration** | `reports/ai_use_declaration.md` | Reproducibility and AI-use (10) | ☐ |

### 1. Problem framing canvas

One page that answers:

- **Stakeholder:** hotel revenue manager.
- **Decision:** which new bookings to follow up (reminder, deposit request, overbooking plan).
- **Primary lens:** cancellation risk. No secondary lens.
- **Unit of analysis:** one booking, scored at the moment it is made.
- **Task:** binary classification (cancelled / not cancelled).
- **Output:** a cancellation probability (0–1) plus a yes/no flag using the chosen threshold (0.335).
- **Why this lens:** cancellations are 27.7% of bookings; knowing which ones are risky lets the hotel act early.
- **Success measure:** PR-AUC (focuses on the cancelled class) and recall — not accuracy.

### 2. Workflow diagram

One picture, left to right:

`Raw data (119,390)` → `Clean + remove duplicates & leaks (01)` → `80/20 split (02)` →
`5 models tuned with cross-validation (03–07)` → `Pick best on training data (08)` →
`Test once (08)` → `Checks: class weighting (09), ensemble (10)` → `Save model (models/)` → `Recommendation`

Mark where the test set is used, so the examiner sees it is kept apart.

### 3a. EDA insight log

A table: **observation → evidence (number/chart) → possible reason → what we did about it**.
Rebuild from notebook 01 and `project_record.md` §3–4, for example: lead time vs cancellation,
City vs Resort hotel, Non Refund deposits, duplicates (37.1% → 27.7% cancel rate), missing values.

### 3b. Preprocessing / feature log

A table: **issue → options → what we chose → evidence**. Cover every item the rubric names:
missing values, outliers, duplicates, categories (one-hot, rare → shared column), scaling,
imbalance (class weighting tested, SMOTE rejected), leakage (dropped post-booking columns, +0.0407 if kept),
new features (`total_nights`, `total_guests`, `is_family`) and the Kaggle features that did not help.
Source: `project_record.md` §2, §4–6 and `reports/results/ablation.csv`.

### 4. Recommendation + limitations

- **What to do:** use XGBoost to flag high-risk bookings; contact those guests, ask for a deposit,
  or plan overbooking around them.
- **What to expect:** catches 77% of cancellations; about 65% of flagged bookings really cancel.
- **Where it is weak:** short-notice bookings (missed cancellations have a median lead time of 42 days
  against 92 for caught ones).
- **Risks and fairness:** `country` is the strongest input — the hotel must not use the flag to treat
  guests from some countries worse (e.g. refusing bookings); use it only for planning and friendly reminders.
- **Limitations:** from `project_record.md` §10 (two Portuguese hotels, 2015–2017, threshold picked
  for F1 not cost, no time-based test).
- **Ensemble:** tried, no real gain → keep the simpler model.

### 5. AI-use declaration

State honestly what AI was used for (drafting code and explanations, reviewing, running checks) and
that the group reviewed and accepted every decision. Base it on `project_record.md` §11.

---

## Part 2 — group work (needs every member)

| # | Task | Who | Output | Done? |
| --- | --- | --- | --- | --- |
| 6 | **Final report** | All (one editor) | Report built from the Part 1 documents + notebooks | ☐ |
| 7 | **Viva preparation** | Each member | Can explain their own notebook in simple words | ☐ |
| 8 | **3-minute YouTube demo video** | All | Unlisted YouTube link in the report | ☐ |
| 9 | **Personal Learning Journey report** | Each member, individually | One A4 page each | ☐ |

### 6. Final report

Follow the rubric order: problem framing → workflow + decisions → data and EDA → preprocessing →
models and comparison → evaluation → recommendation and limitations → reproducibility and AI use.
Every number must match the notebook outputs. Include the dataset link and fingerprint (`data/raw/SOURCE.md`)
and the data dictionary (`reports/data_dictionary.csv`).

### 7. Viva preparation

Suggested split (confirm names):

| Member | Notebooks to explain |
| --- | --- |
| M1 | 01 EDA and cleaning, 02 preprocessing, 03 Logistic Regression |
| M2 | 04 Decision Tree |
| M3 | 05 Random Forest |
| M4 | 06 XGBoost |
| Shared | 07 Neural Network, 08 model selection, 09 class weighting, 10 ensemble |

Everyone should also be able to answer these in one sentence each:

- Why PR-AUC and not accuracy?
- Why did we remove duplicates and the post-booking columns?
- Why is the test set used only once?
- Why is ~0.76 the ceiling, and why do Kaggle notebooks claim 97–99%?
- Why did we keep XGBoost instead of the ensemble?

A one-page "how to explain it" sheet per notebook can be prepared for each member.

### 8. 3-minute demo video

Rough script: problem (20 s) → data and cleaning (30 s) → models compared (40 s) →
final model and test result (30 s) → recommendation for the hotel (30 s) → limitations (20 s) →
close (10 s). Upload to YouTube as unlisted and put the link in the report.

### 9. Personal Learning Journey report (individual)

One A4 page, written by each member in their own words: what I did, what I learned, what was hard,
what I would do differently. This is marked individually, so it must be personal.

---

## Deadline

**Check the real submission deadline** (the old plan assumed 24 September, which has passed).

## How to run the notebooks

Use the project `.venv` (it has xgboost). In VS Code / Jupyter pick the `.venv` interpreter or the
"CampusML (.venv)" kernel. Run the notebooks in order 01 → 10; 02 must run before any model notebook.
