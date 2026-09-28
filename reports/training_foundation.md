# Training foundation (v2)

v2 keeps the v1 population and predictors and changes only how rows are split: rows with the same
booking profile always stay together. Everything described here exists in the working tree as of
21 September 2026. The v1 files (`notebooks/v1/`, `data/processed/train.csv`, `data/processed/test.csv`,
`reports/results/`) stay as they are and are historical.

## Files

| Step | Notebook | Output |
| --- | --- | --- |
| EDA and cleaning (shared with v1) | `notebooks/01_eda_cleaning.ipynb` | `data/processed/hotel_bookings_clean.csv` |
| Group-aware train/test split | `notebooks/v2/02_preprocessing_v2.ipynb` | `data/processed/v2/train.csv`, `data/processed/v2/test.csv`, `data/processed/v2/manifest.json` |
| Nested cross-validation folds | `notebooks/v2/03_cv_folds_v2.ipynb` | `data/processed/v2/cv_folds.csv`, adds to `manifest.json` |

## Contract

- **Population:** the v1 cleaned table with duplicate rows removed (notebook 01), 84,969 rows. It is a
  "unique profiles" population, not "all recorded bookings".
- **Target:** `is_canceled` (1 cancelled, 0 not).
- **Predictors:** the 25 columns of `train.csv` other than `is_canceled`. `arrival_date_year` and
  `arrival_date_week_number` are dropped. `agent` is an ID and is read as text:
  `pd.read_csv(..., dtype={'agent': str})`.
- **Group key (profile):** an exact match on all 25 predictors, with the target left out.
  In code: `pd.util.hash_pandas_object(X, index=False).factorize()[0]`.
- **Split:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, fold 0 = test.
  Train 67,974 rows and test 16,995 rows, cancellation rate 0.2775 on both. Profiles in both train and
  test: 93 in v1, 0 in v2.
- **CV folds** (`cv_folds.csv`, one row per training row, in `train.csv` order):
  - `row_id`: 0-based row position in `train.csv`.
  - `outer_fold` (0–4): `StratifiedGroupKFold(5, shuffle=True, random_state=42)` on all training rows.
  - `inner_fold_0` … `inner_fold_4` (0–2): for outer fold k, `StratifiedGroupKFold(3, shuffle=True,
    random_state=42)` on the rows outside it. `inner_fold_k` is blank on rows whose `outer_fold` is k.
  - Every outer and inner fitting/validation pair has 0 shared profiles. Cancellation rates on
    every side range from 0.2774 to 0.2775.
- **Manifest** (`data/processed/v2/manifest.json`): source file and its SHA-256, population, dropped
  columns, target, group key, splitter, seed, row counts, cancellation rates, profile overlap, the
  `cv_folds` settings, and a SHA-256 for `train.csv`, `test.csv` and `cv_folds.csv`.

## Reproduce

Run the three notebooks in order with the project `.venv` Python. The default `python3` Jupyter
kernel may run system Python instead, so register a kernel that points at `.venv`:

```powershell
.venv\Scripts\python.exe -m ipykernel install --user --name campusml-venv
.venv\Scripts\jupyter-nbconvert.exe --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=campusml-venv notebooks\01_eda_cleaning.ipynb
.venv\Scripts\jupyter-nbconvert.exe --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=campusml-venv notebooks\v2\02_preprocessing_v2.ipynb
.venv\Scripts\jupyter-nbconvert.exe --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=campusml-venv notebooks\v2\03_cv_folds_v2.ipynb
```

Notebook 02 rewrites `manifest.json` without the `cv_folds` entry, and notebook 03 adds it back, so
always run 03 after 02. On 21 September 2026 this sequence reproduced every hash in the manifest, and
a second run of 03 gave the same `cv_folds.csv` hash. That run used the existing `.venv`; an install
into a fresh environment from `requirements.txt` has not been tested.

## Using the folds in a model notebook

```python
import pandas as pd

train = pd.read_csv('../../data/processed/v2/train.csv', dtype={'agent': str})
folds = pd.read_csv('../../data/processed/v2/cv_folds.csv', dtype='Int64')
X, y = train.drop(columns='is_canceled'), train['is_canceled']

for k in range(5):
    fit = folds.outer_fold != k
    inner = folds.loc[fit, f'inner_fold_{k}']
    inner_cv = [((inner != j).to_numpy().nonzero()[0], (inner == j).to_numpy().nonzero()[0])
                for j in range(3)]
    # tune on X[fit] with cv=inner_cv, then score X[~fit]; save out-of-fold probabilities by row_id
```

Inner indices are positions within `X[fit]`. Do not swap the saved folds for ordinary row-stratified
folds, and never use `row_id` or the profile key as a predictor.

## Boundaries

- No v2 model has been trained yet. Retraining the models on these folds is plan.md §9.3.
- v1 scores come from a row-stratified split that had 93 shared profiles; do not compare them with
  v2 scores.
- `test.csv` is opened once, for the final frozen model (plan.md §9.7). No decision uses it before then.
- The test set is historical data from the same period, not a future season, so no claim about
  performance on future bookings is made.
