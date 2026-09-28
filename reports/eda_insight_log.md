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
| 2 | **Missing values in four columns** | `company` 94.3% empty, `agent` 13.7%, `country` 0.41% (488 rows), `children` 4 rows (§2). The most common country, PRT, cancels 56.6%, but rows with no country cancel only 13.7% (§2) | `agent` is empty when the guest booked without a travel agent (46.7% of those are Direct, 33.9% Corporate); missing-country bookings are a different kind of booking | `company` dropped; `agent` → 0 ("no agent"); `country` → its own `Unknown` level, not the mode (that would label them as the highest-cancelling country); `children` → 0 |
| 3 | **Impossible values** | 180 bookings with zero guests; one price of −6.38 and one of 5,400 (next highest is 510) (§3) | Data entry errors | Removed only these 182 rows. We did **not** cut outliers by percentile: expensive peak-season bookings are real |
| 4 | **A column that is "too good": parking** | 7,409 bookings asked for parking and **0** were cancelled; about 2,747 would be expected (§4.1) | Parking is recorded when the guest arrives, so it really means "the guest turned up" | Dropped as a leak with the other post-booking columns (`reservation_status*`, `assigned_room_type`, `booking_changes`, `days_in_waiting_list`). Keeping them would add **+0.0407** PR-AUC (`ablation.csv` arm 6) |
| 5 | **Two time columns mislead** | 2015 has 6 months of arrivals and 2017 has 8, so the cancel rate "rises" 20.8% → 26.6% → 32.1%; week number vs month number correlation = **0.995** (notebook 02 §2) | The yearly trend partly comes from which months each year covers, and future bookings fall in unseen years; week number and month describe the same thing | Dropped `arrival_date_year` and `arrival_date_week_number`; kept month + day of month |

## Cancellation patterns

| # | Observation | Evidence (number / chart, where) | Possible reason | What we did about it |
| --- | --- | --- | --- | --- |
| 6 | **Classes are imbalanced** | 27.7% cancelled; always saying "not cancelled" is 72.3% accurate (§6.1, pie chart) | Most guests do arrive | Ranked models on **PR-AUC**, reported recall, stratified the split, and chose a decision threshold instead of 0.5 |
| 7 | **Longer lead time → more cancellations** | 16.5% (0–30 days) → 31.9% (31–60) → 35.3% (91–180) → 40.8% (181–365) → 42.9% (366+); median 79 days for cancelled vs 37 for kept (§6.3, bar chart + density plot) | A booking made a year ahead is cheap to abandon; a booking for next week is a firm plan | Kept `lead_time` as a number. It is the 2nd most important input (permutation importance 0.137, `feature_importance.csv`) |
| 8 | **City Hotel cancels more than Resort Hotel** | 30.3% vs 23.7% (§6.2, bar chart) | Different guests: city trips are easier to change than holidays | Kept `hotel` as a feature |
| 9 | **Non Refund deposits almost always cancel, but are rare** | 94.9% cancel, but only 990 of 84,969 bookings (§6.4) | Unclear from the data (a guest who paid in full cancelling is surprising); the group is tiny, so the rate is not a general rule | Kept `deposit_type`. Removing it costs only −0.0040 PR-AUC (`ablation.csv` arm 13): a big headline, small effect |
| 10 | **Online travel agents are the riskiest large segment** | Online TA 35.3% (51,081 bookings) vs Offline TA/TO 15.0%, Direct 14.7%, Corporate 12.1% (§6.5) | Online bookings are easy to make and cancel, often with free cancellation | Kept `market_segment` (one-hot) |
| 11 | **Special requests → fewer cancellations** | 33.8% with no request, down to 10.8% with four (§6.6); holds inside every lead-time band (§4.2) | A guest who asks for something (e.g. high floor) is planning to come | Kept. Removing it costs **−0.0530** PR-AUC, the biggest single-feature effect (`ablation.csv` arm 14) |
| 12 | **Country: home guests cancel more** | 88.5% of guests are European; Portuguese (domestic) 35.9% vs rest of Europe 22.5%; PRT has 26,439 bookings (§6.8, §6.10) | Travelling within your own country is cheaper to cancel than a trip with flights | Kept `country`; the 142 countries with fewer than 100 bookings share one column. It is the strongest input, which raises a **fairness** point, see [recommendation.md](recommendation.md) |

## Checked on training rows only

The charts above were drawn on all 84,969 rows, which include the later test rows. Notebook 02 §5.1 redraws
the four charts that drove decisions (lead time, deposit type, market segment, special requests) on the
**67,975 training rows only**. Every large group moves by **less than half a percentage point**; only tiny groups
move more (Refundable −3.25 points, Aviation −2.27). So no decision depended on seeing the test data.

**These are associations, not causes.** For example, lead time does not *make* a guest cancel; it is a sign of
the kind of booking.
