# Feature engineering and preprocessing decisions — v2

These choices were specified before measuring the revised diagnostics. The target was not used to compute engineered values. Each formula uses only the same row; learned encoders, imputers and scalers remain inside training pipelines. Initial-booking availability of the retained snapshot values is still unverified.

| Feature / option | Definition | Justification | Decision and caveat |
| --- | --- | --- | --- |
| total_nights | weekday + weekend nights | An interpretable total duration | Replace weekday nights; keep weekend nights. Invertible representation, no new information, no exact total-plus-all-components dependency. |
| total_guests | adults + children + babies | Interpretable party size | Replace adults; keep children and babies. Preserves information but depends on child-zero approximation. |
| booked_by_company | company ID is not None | Source-backed presence of associated organisation | Keep a compact business descriptor; company identity is omitted by design, not declared useless. Predictive gain is not established. |
| is_family | children > 0 or babies > 0 | Nonlinear summary of composition | Candidate only: very small mixed tree effects do not justify a default extra column. |
| children_missing | original count absent | Preserve uncertainty about assumed zeros | Audit/candidate only: four raw records; no gain in the fixed-tree experiment. |
| agent presence | agent ID is not None | Intermediary involvement | No extra flag: categorical agent already represents None. |
| prior cancellation ratio | prior cancellations / all prior bookings | Potential history summary | Deferred; zero history is not a measured 0% rate, small denominators are unstable, and history timing must be verified. |
| ADR per person | rate / party size | Proposed normalisation | Not adopted: ADR is per room-night; room count is unknown and ADR timing uncertain. |
| lead-time bands / arrival quarter | fixed coarse bins | Readable plots | EDA-only; preserve original detail for modelling. |
| room_changed | assigned room differs from reserved | Allocation history | Excluded: derives from an input with uncertain operational timing. |
| year / week | arrival calendar fields | Cohort auditing / additional season detail | Audit only. Partial years complicate comparisons; month/day suffice for the default representation. Not proven useless. |

## What the training-only evidence actually says

All arms use the same five group-aware validation populations and a fixed shallow tree (depth 8, minimum leaf 50, seed 42). No tuning or test evaluation occurs. Mean AP and sample fold standard deviation:

| arm | mean | std |
| --- | --- | --- |
| candidate_children_indicator | 0.81541 | 0.00981 |
| candidate_plus_family | 0.81540 | 0.00983 |
| default_plus_company | 0.81541 | 0.00981 |
| exact_raw_dedup_training_only | 0.79893 | 0.00408 |
| raw_components | 0.81669 | 0.00919 |
| totals_replacing_components | 0.81541 | 0.00981 |

Paired AP changes relative to the default totals-plus-company representation:

| arm | mean | min | max |
| --- | --- | --- | --- |
| candidate_children_indicator | 0.0000000 | 0.0000000 | 0.0000000 |
| candidate_plus_family | -0.0000102 | -0.0000895 | 0.0000403 |
| default_plus_company | 0.0000000 | 0.0000000 | 0.0000000 |
| exact_raw_dedup_training_only | -0.0164716 | -0.0286034 | -0.0077307 |
| raw_components | 0.0012798 | 0.0004313 | 0.0027022 |
| totals_replacing_components | -0.0000005 | -0.0000022 | 0.0000000 |

Raw components perform slightly better than totals for this tree across these folds. Totals are retained as a documented, information-preserving representation choice, **not a demonstrated predictive improvement**. Company presence changes AP by less than a few millionths; its strong univariate association adds essentially no incremental tree evidence here. Retaining its one compact business indicator is a domain choice, not a performance claim. Neither choice is declared optimal for future models.

The family flag has tiny mixed changes; the child-missing flag changes none of these scores. They remain outside the default contract. There are too few missing-child examples to establish that the zero assumption is correct.

Removing exact raw duplicates from each training fit reduces this fixed tree's AP on the same booking-weighted validation rows. This supports preserving the stated all-bookings population for this experiment; it does not prove that duplicates are genuine bookings or quantify memorisation. There is no profile overlap across folds in either arm.

Fold SD is not a significance test. These are exploratory diagnostics for one estimator, not a final multi-model nested-CV result. Close results must not be advertised as a proven winner. No additional interactions or bins were added in response to test performance.

## Fitted preprocessing

Nine categorical columns use one-hot encoding with a minimum training frequency of 100. An unseen value uses the fitted infrequent bucket if one exists, otherwise the corresponding block is zero. Eleven numeric columns use a training-fitted median fallback and optional StandardScaler. Current cleaned predictors have no missing values, so the numeric fallback does not change them. The synthetic missing/unknown probes and scaler checks passed.

The same builder is used for every diagnostic and is the required handoff to future model notebooks. The earlier whole-frame frequency encoder is not used; its old score is not valid evidence that one-hot is superior.
