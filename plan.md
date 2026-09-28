# IT3091 ML — plan

**Group 2026-AI-46 · Guided Data Track, code 6 (Hotel Booking Demand) · predicting booking cancellations.**

Numbers and decisions: [`reports/project_record.md`](reports/project_record.md) and
[`reports/decision_log.md`](reports/decision_log.md).

## Where we are (28 September 2026)

The ML workflow is complete: **collect data → clean → train and compare models → pick one → save it.**

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
All honest approaches land at about 0.76–0.77 PR-AUC; that is the ceiling of the data.

## Still to do

1. **Viva preparation.** Each member must explain their own notebook in simple words. Suggested split
   (confirm names): M1 notebooks 01–03, M2 04, M3 05, M4 06; shared: 07–10.
2. **Final report.**
3. **3-minute video.**
4. **Personal learning reports.**
5. **AI-use declaration** (see project_record.md §11).
6. **Check the real submission deadline.**

## How to run the notebooks

Use the project `.venv` (it has xgboost). In VS Code / Jupyter pick the `.venv` interpreter or the
"CampusML (.venv)" kernel. Run the notebooks in order 01 → 10; 02 must run before any model notebook.
