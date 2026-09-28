# EDA insight log

What we saw in the data, the evidence, a likely reason, and what we did about it.
Unless another place is named, the numbers come from `notebooks/01_eda_cleaning.ipynb` (section numbers in brackets).
Rates after cleaning are on the 84,969 deduplicated bookings.

**One row = one booking.** Target: `is_canceled` (1 = cancelled or no-show, 0 = stayed).
Column meanings and types: [data_dictionary.csv](data_dictionary.csv).

## Data quality

| # | Observation | Evidence (number / chart, where) | Possible reason | What we did about it |
| --- | --- | --- | --- | --- |
| 1 | **Many exact duplicate rows** | 34,239 rows are exact copies of another row; 55.9% of them are cancellations (§5) | No booking ID in the data, so group bookings or system exports repeat identical rows | Removed them. Cancel rate moves from **37.1% to 27.7%**. Keeping them gives PR-AUC **0.9225** vs **0.7660**, because the model memorises copies that land in both train and test (`ablation.csv` arm 7) |
| 2 | **Missing values in four columns** | `company` 94.3% empty, `agent` 13.7%, `country` 0.41% (488 rows), `children` 4 rows (§2) | `agent` is empty when the guest booked without a travel agent: 46.7% of those are Direct and 33.9% Corporate (§2) | `company` dropped; `agent` → 0 ("no agent"); `country` → own `Unknown` level (not the mode); `children` → 0 |
| 3 | **Filling `country` with the most common value would be wrong** | PRT (the mode, 40.9% of rows) cancels 56.6%, but rows with no country cancel only 13.7% (§2) | Missing-country bookings are a different kind of booking | Gave them their own `Unknown` category instead of "PRT" |
| 4 | **Impossible values** | 180 bookings with zero guests; one price of −6.38 and one of 5,400 (next highest is 510) (§3) | Data entry errors | Removed only these 182 rows. We did **not** cut outliers by percentile: expensive peak-season bookings are real |
| 5 | **A column that is "too good": parking** | 7,409 bookings asked for parking and **0** were cancelled; about 2,747 would be expected (§4.1) | Parking is recorded when the guest arrives, so it really means "the guest turned up" | Dropped as a leak with the other post-booking columns (`reservation_status*`, `assigned_room_type`, `booking_changes`, `days_in_waiting_list`). Keeping them would add **+0.0407** PR-AUC (`ablation.csv` arm 6) |
| 6 | **Partial years** | 2015 has 6 months of arrivals, 2017 has 8; the cancel rate "rises" 20.8% → 26.6% → 32.1% (notebook 02 §2) | The trend partly comes from which months each year covers; future bookings will be in years the model never saw | Dropped `arrival_date_year` |
| 7 | **Week number repeats the month** | Correlation of week number with month number = **0.995** (notebook 02 §2) | Both describe the same position in the year | Dropped `arrival_date_week_number`; kept month + day of month |

## Cancellation patterns

| # | Observation | Evidence (number / chart, where) | Possible reason | What we did about it |
| --- | --- | --- | --- | --- |
| 8 | **Classes are imbalanced** | 27.7% cancelled; always saying "not cancelled" is 72.3% accurate (§6.1, pie chart) | Most guests do arrive | Ranked models on **PR-AUC**, reported recall, stratified the split, and chose a decision threshold instead of 0.5 |
| 9 | **Longer lead time → more cancellations** | 16.5% (0–30 days) → 31.9% (31–60) → 35.3% (91–180) → 40.8% (181–365) → 42.9% (366+); median 79 days for cancelled vs 37 for kept (§6.3, bar chart + density plot) | A booking made a year ahead is cheap to abandon; a booking for next week is a firm plan | Kept `lead_time` as a number. It is the 2nd most important input (permutation importance 0.137, `feature_importance.csv`) |
| 10 | **City Hotel cancels more than Resort Hotel** | 30.3% vs 23.7% (§6.2, bar chart) | Different guests: city trips are easier to change than holidays | Kept `hotel` as a feature |
| 11 | **Non Refund deposits almost always cancel, but are rare** | 94.9% cancel, but only 990 of 84,969 bookings (§6.4) | Probably a data-recording quirk or group blocks; the group is tiny | Kept `deposit_type`. Removing it costs only −0.0040 PR-AUC (`ablation.csv` arm 13): a big headline, small effect |
| 12 | **Online travel agents are the riskiest large segment** | Online TA 35.3% (51,081 bookings) vs Offline TA/TO 15.0%, Direct 14.7%, Corporate 12.1% (§6.5) | Online bookings are easy to make and cancel, often with free cancellation | Kept `market_segment` (one-hot) |
| 13 | **Special requests → fewer cancellations** | 33.8% with no request, down to 10.8% with four (§6.6); holds inside every lead-time band (§4.2) | A guest who asks for something (e.g. high floor) is planning to come | Kept. Removing it costs **−0.0530** PR-AUC, the biggest single-feature effect (`ablation.csv` arm 14) |
| 14 | **Guest history matters** | Previous cancellation → 67.3% cancel again (1,647 bookings); repeat guests 7.7% (§6.6) | Past behaviour predicts future behaviour | Kept `previous_cancellations`, `previous_bookings_not_canceled`, `is_repeated_guest` |
| 15 | **Country: home guests cancel more** | 88.5% of guests are European; Portuguese (domestic) 35.9% vs rest of Europe 22.5%; PRT has 26,439 bookings (§6.8, §6.10) | Travelling within your own country is cheaper to cancel than a trip with flights | Kept `country`; the 142 countries with fewer than 100 bookings share one column. It is the strongest input, which raises a **fairness** point, see [recommendation.md](recommendation.md) |
| 16 | **Seasonality** | Highest in August (32.4%) and July (31.9%), lowest in November (21.4%); Resort prices go from 49 in Jan to 188 in Aug (§6.7) | Summer holidays are booked early and changed more | Kept `arrival_date_month` and `adr` |
| 17 | **No single numeric column explains cancellation** | Largest correlations: `lead_time` 0.19, `adr` 0.13, `total_of_special_requests` −0.13 (§6.9, heatmap) | Cancellation depends on a combination of factors | Used models that can combine many inputs (trees, boosting, neural network), not just a linear model |

## Checked on training rows only

The charts above were drawn on all 84,969 rows, which include the later test rows. Notebook 02 §5.1 redraws
the four charts that drove decisions (lead time, deposit type, market segment, special requests) on the
**67,975 training rows only**. Every large group moves by **less than half a percentage point**; only tiny groups
move more (Refundable −3.25 points, Aviation −2.27). So no decision depended on seeing the test data.

**These are associations, not causes.** For example, lead time does not *make* a guest cancel; it is a sign of
the kind of booking.
