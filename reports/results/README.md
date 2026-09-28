# Model results

CSV outputs of the notebooks. Every model is trained on `data/processed/train.csv` (67,975 bookings, 25 features).

| File | Written by |
| --- | --- |
| `logistic_regression.csv`, `decision_tree.csv`, `random_forest.csv`, `xgboost.csv`, `neural_network.csv` | notebooks 03–07: CV score, chosen settings, threshold |
| `leaderboard.csv`, `final_model.csv` | notebook 08: ranking and the final XGBoost test result |
| `weighted_comparison.csv`, `weighted_models.csv`, `weight_sweep.csv` | notebook 09 and `scripts/weight_sweep.py`: class weighting check |
| `ensemble.csv` | notebook 10: soft-voting ensemble against XGBoost |
| `ablation.csv`, `feature_importance.csv` | `scripts/ablation.py`, `scripts/feature_importance_check.py`: feature tests |

The trained models themselves are in `models/` at the project root.
