# AI-use declaration

**Group 2026-AI-46 · IT3091 Machine Learning · Guided Data Track, code 6 (Hotel Booking Demand).**

We used an AI assistant during this project. This page says honestly what it did, what we did, and how we
checked its work. Based on [project_record.md](project_record.md) §11.

## Tool used

| Tool | How we used it |
| --- | --- |
| **Claude** (Anthropic), through Claude Code | A coding assistant working inside our project folder: it could read our files, write code and text, and run our notebooks and scripts on our machine |

## What the AI helped with

| Area | What the AI did |
| --- | --- |
| **Code** | Drafted code for the notebooks (01–10), the check scripts in `scripts/` (`ablation.py`, `feature_importance_check.py`, `weight_sweep.py`, `weighted_comparison.py`, `make_workflow_diagram.py`) and the demo app `app/app.py` |
| **Notebook text** | Drafted the markdown explanations under the charts and tables |
| **Reviewing decisions** | Reviewed the shared foundation (cleaning, split, encoding) and pointed out risks, for example leak columns and duplicate rows |
| **Running checks** | Ran the feature tests (`ablation.py`) and the importance check (`feature_importance_check.py`) and reported the numbers |
| **Documents** | Drafted the evidence documents in `reports/` (this file, the problem framing canvas, workflow diagram, EDA insight log, preprocessing/feature log, recommendation, decision log and project record) from the notebook outputs |

## What the group did

As planned in the initial submission (section 5.2), each member owned one part of the data work
(notebooks 01 and 02) and one model, and is the person who explains that part.

| Member | Name (student ID) | What they did |
| --- | --- | --- |
| M1 (project lead) | Warnakulasinhage S.N.A (IT24101147) | Notebook 02 in full (features and target, encoding plan, stratified 80/20 split, 5 cross-validation folds, PR-AUC metric); Logistic Regression (03, baseline); coordination: workflow diagram, decision log, model comparison (08), report assembly and the demo video |
| M2 | Nawodya K.P.G.P (IT24102629) | Notebook 01 sections 1–3 and 5 (loading, data dictionary, missing values, impossible rows and outliers, duplicates); Decision Tree (04) |
| M3 | Fonseka W. P. L (IT24100509) | Notebook 01 sections 4 and 6.1–6.6 (leakage check, class balance, EDA on booking drivers); Random Forest (05) |
| M4 | Seelarathna G.P.B (IT24101027) | Notebook 01 sections 6.7–9 (seasonality, country, price, correlation, feature engineering, stress tests, saving the clean data); XGBoost (06) |
| All | | Initial submission; Neural Network (07), model selection (08), class weighting (09), ensemble (10); final report, demo video, viva |

The Neural Network and the ensemble were added beyond the four planned models.

## How we checked the AI's work

- **Every decision was reviewed and accepted by a person** before it went into the project. The AI suggested;
  we decided. Each decision is recorded with its options and evidence in [decision_log.md](decision_log.md).
- **Numbers come from running the code, not from the AI's text.** Every number in the reports is taken from a
  notebook output or a CSV in `reports/results/`, and can be checked by re-running the notebooks in order 01 → 10.
- **Tested instead of trusted.** When a change was suggested (extra features, class weighting, an ensemble), we
  measured it and kept it only if it beat the noise between folds. Most suggestions did not, and were rejected.
- **Popular answers were checked too.** Kaggle notebooks report 97–99% accuracy on this dataset. We measured
  why (leak columns and duplicate rows) instead of copying their approach.

## Other sources

Kaggle notebooks consulted for structure (their features were measured, not copied; together +0.0026 PR-AUC):

- [Hotel Booking Cancellation Prediction / EDA + 6 ML — Dalileh Oladzadeh](https://www.kaggle.com/code/dalileholadzadeh/hotel-booking-cancellation-prediction-eda-6-ml)
- [Hotel Cancellation Prediction using ANN — Aamir](https://www.kaggle.com/code/aamir5659/hotel-cancellation-prediction-using-ann)
- [HotelBookingCancellationAnalysis-V1 — Kushal](https://www.kaggle.com/code/kushal1147/hotelbookingcancellationanalysis-v1)
- [Hotel Booking — Hazem Alanany](https://www.kaggle.com/code/hazemalanany/hotel-booking)
- [Hotel Booking Cancellation Analysis / Python — Ahmed Baqa](https://www.kaggle.com/code/ahmedbaqa/hotel-booking-cancellation-analysis-python)

Dataset: [Hotel Booking Demand (Kaggle)](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand),
from Antonio, Almeida and Nunes, *Data in Brief* 22 (2019). Local copy and SHA-256 fingerprint: `data/raw/SOURCE.md`.

## Declaration

We confirm that this declaration is accurate, that we understand every part of the submitted work, and that
each member can explain their own part without AI help.

| Member | Name | Signature / date |
| --- | --- | --- |
| M1 | Warnakulasinhage S.N.A (IT24101147) | |
| M2 | Nawodya K.P.G.P (IT24102629) | |
| M3 | Fonseka W. P. L (IT24100509) | |
| M4 | Seelarathna G.P.B (IT24101027) | |
