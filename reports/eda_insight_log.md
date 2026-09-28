# EDA insight log — revised foundation v2

Run date: 17 September 2026. Full-data entries below describe fixed quality/scope rules. Outcome associations use training rows only. The source is two historical hotels, not a representative sample of the hospitality industry.

| Observation | Supporting count / artifact | Possible explanation | Modelling consequence | Uncertainty |
| --- | --- | --- | --- | --- |
| Some reservations have no guest or no overnight stay | 180 zero-guest removals; 645 additional zero-night removals; cleaning_audit.csv | Administrative/day-use/group records | Define an overnight, at-least-one-guest scope; preserve rejected rows | They are not proven errors; this changes the prediction population |
| Zero-night raw count exceeds sequential removals | 715 raw zero-night records; 70 also have zero guests | Overlapping scope conditions | Apply one sequential reason per rejection | Do not double-count exclusions |
| Rare unknown child count | Four raw missing values | Unknown export value | Fixed zero approximation, audit indicator and sensitivity | No evidence that every missing value really means zero |
| Agent/company blanks are common | missingness_by_group.csv | Source says not applicable for these ID categories | Explicit None agent; company presence flag | Meaning is source-based, not inferred only from channels |
| Prices/parties have unusual values | quality_flags.csv and flagged_rows.csv | Complimentary/group/refund records or possible error | Keep; flag; ADR stays audit-only | Magnitude alone does not prove data error |
| Many exact records/profile matches | 31,994 raw duplicate excess; 37,460 repeated final X profiles after scope cleaning | Legitimate repeated bookings or duplicated records | Preserve weights, group across every split | No booking IDs to distinguish explanations |
| Same observed profile can have different outcomes | 571 coarser groups; 3,918 rows in those groups | Inputs cannot fully describe cancellation behaviour | Retain all labels in same split/fold | Not proof of label corruption |
| Incomplete year coverage | figures/02_partial_year_coverage.png | Collection starts July 2015, ends August 2017 | Avoid annual growth claims; keep year for audit | Monthly mix still varies by hotel/year |
| Cancellation differs by hotel | train_rate_hotel.csv | Channel/customer/policy mix | Keep hotel identity as a contextual descriptor | Association is not a causal hotel effect |
| Cancellation increases across lead bands | train_rate_lead_time_band.csv | Longer exposure to plan changes is one hypothesis | Keep continuous lead_time; bands are EDA-only | Mechanism unverified; historical snapshot limitation |
| Company-associated bookings have different rates | train_rate_booked_by_company.csv | Corporate/channel mix | Compact flag; assess incremental value, not just rates | Fixed-tree ablation finds essentially no gain |
| Deposit/request/parking patterns are strong | train_rate_deposit_type.csv; train_rate_total_of_special_requests.csv; train_rate_required_car_parking_spaces.csv | Policy, selection and update processes | Audit and conservative exclusion | Rate patterns do not establish entry timestamps |

## Training observations, with denominators and descriptive intervals

Hotel:

| hotel | n | cancellations | rate | lower | upper |
| --- | --- | --- | --- | --- | --- |
| City Hotel | 63073 | 26528 | 0.42059 | 0.41674 | 0.42445 |
| Resort Hotel | 31789 | 8818 | 0.27739 | 0.27250 | 0.28234 |

Lead time:

| lead_time_band | n | cancellations | rate | lower | upper |
| --- | --- | --- | --- | --- | --- |
| 0-30 | 30481 | 5738 | 0.18825 | 0.18390 | 0.19268 |
| 31-90 | 23510 | 8826 | 0.37541 | 0.36925 | 0.38162 |
| 91-180 | 21209 | 9505 | 0.44816 | 0.44148 | 0.45486 |
| 181-365 | 17170 | 9557 | 0.55661 | 0.54917 | 0.56403 |
| 366+ | 2492 | 1720 | 0.69021 | 0.67177 | 0.70806 |

Company association:

| booked_by_company | n | cancellations | rate | lower | upper |
| --- | --- | --- | --- | --- | --- |
| 0 | 89513 | 34409 | 0.38440 | 0.38122 | 0.38759 |
| 1 | 5349 | 937 | 0.17517 | 0.16522 | 0.18559 |

Deposit type (audit only):

| deposit_type | n | cancellations | rate | lower | upper |
| --- | --- | --- | --- | --- | --- |
| No Deposit | 83220 | 23867 | 0.28679 | 0.28373 | 0.28988 |
| Non Refund | 11534 | 11451 | 0.99280 | 0.99109 | 0.99419 |
| Refundable | 108 | 28 | 0.25926 | 0.18589 | 0.34916 |

Wilson 95% intervals assume independent Bernoulli bookings. Related and repeated records can make them too narrow; they are descriptive, not cluster-adjusted inferential results. Full country/agent/request tables retain rare groups; plots show frequent countries/agents for readability.

## Model dependence, not causal explanation

Training-slice permutation importance for the fixed diagnostic tree:

| feature | AP_decrease | repeat_std |
| --- | --- | --- |
| country | 0.24140 | 0.00151 |
| lead_time | 0.11281 | 0.00257 |
| agent | 0.08399 | 0.00151 |
| customer_type | 0.07297 | 0.00175 |
| market_segment | 0.04861 | 0.00275 |
| previous_cancellations | 0.03616 | 0.00258 |
| previous_bookings_not_canceled | 0.00853 | 0.00019 |
| distribution_channel | 0.00824 | 0.00088 |

Correlated substitutes can reduce measured importance; independent shuffling can also create implausible combinations. Repeat variability is not full sampling uncertainty. This ranking did not remove any predictors. No statement about country establishes nationality, travel motivation or an inherent personal tendency to cancel.
