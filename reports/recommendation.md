# Recommendation and limitations

**For:** the hotel revenue manager. **Question:** which new bookings should we follow up because they are
likely to be cancelled?

All numbers are from the test set (16,994 bookings the model never saw during training), notebook 08,
unless another source is named.

## 1. What the hotel should do

**Use the XGBoost model (`models/xgboost.joblib`) to score every new booking when it is made.** If the
cancellation probability is **0.335 or higher**, flag the booking as *high risk*. For flagged bookings:

| Action | Why it fits |
| --- | --- |
| **Friendly reminder** before arrival (email / message, confirm the dates) | Cheap, polite, and harmless if the flag is wrong |
| **Ask for a deposit or offer a small incentive to confirm**, where the hotel's policy already allows it | Turns a loose booking into a firmer one |
| **Plan overbooking around the flagged total**, e.g. "we expect about X of this week's flagged bookings to cancel" | Uses the flags as numbers for planning, not as judgements about one guest |

Use it as **decision support**: a person (the revenue manager) decides what to do; the model only ranks bookings.

## 2. What to expect

| Result on the test set | Value | In plain words |
| --- | ---: | --- |
| Recall | **0.769** | Catches **about 3 out of 4** real cancellations (77%) |
| Precision | **0.645** | About **65% of flagged bookings really cancel**; about 1 in 3 flags is a false alarm |
| PR-AUC | **0.770** | Ranking quality; a model that guesses scores 0.278 |
| ROC-AUC | **0.892** | A random cancelled booking gets a higher score than a random kept one 89% of the time |
| Accuracy | **0.819** | Shown for completeness only: always saying "not cancelled" already gets 72% |

**Per 1,000 new bookings** (from the confusion matrix TN 10,286 · FP 1,993 · FN 1,088 · TP 3,627,
scaled from 16,994 to 1,000):

- about **277** will cancel;
- the model flags about **331**: **213** real cancellations and **117** false alarms;
- it misses about **64** cancellations.

So about 16% of guests who *will* come get a reminder they did not need (1,993 of 12,279). That is why
the actions above are friendly ones.

The test score (0.7698) landed inside the cross-validation estimate made on training data only
(0.7612 ± 0.0090), so this is a realistic expectation, not a lucky number.

## 3. Where the model is weak

**Short-notice bookings.** The cancellations it misses were booked a median of **42 days** ahead, against
**92 days** for the ones it catches. Missed cancellations are also more often **Direct** bookings (20.9% of
missed vs 3.8% of caught; notebook 08 §5). At booking time, a short-notice booking that will later be cancelled
looks almost the same as one that will be kept. **The hotel should not treat "not flagged" as "safe"**,
especially for last-minute and direct bookings.

## 4. Risks and fairness

- **`country` is the strongest input** (permutation importance 0.173, `feature_importance.csv`). Portuguese
  guests cancel more (35.9%) than other Europeans (22.5%), probably because a domestic trip is cheaper to change.
  **The hotel must not use the flag to treat guests from some countries worse**: no refusing bookings, no
  worse prices, no harder conditions based on the flag. Use it only for planning and friendly reminders.
- **A flag is a probability, not a fact.** About 1 in 3 flags is wrong. Harsh actions (cancelling a
  booking, charging a fee) on a flag alone would hurt real guests.
- **Personal data.** The model uses booking details, not names. Any real use must follow the hotel's data
  protection rules (e.g. GDPR in Portugal).
- **Drift.** Travel habits changed a lot after 2020. A model trained on 2015–2017 may be less accurate today.

## 5. Why the simpler XGBoost and not the ensemble

We also tried averaging Random Forest, XGBoost and the Neural Network (soft-voting ensemble, notebook 10):

| | XGBoost alone | Ensemble |
| --- | ---: | ---: |
| PR-AUC, 5 training folds | 0.7660 | 0.7696 |
| PR-AUC, test set | 0.7698 | 0.7694 |
| F1, test set | 0.7019 | 0.7019 |
| File size | 3 MB | 52 MB |

The ensemble's gain (+0.0037) is **less than half** of XGBoost's normal fold-to-fold wobble (0.0080), and on
the test set there is no gain at all. The three models' predictions are already 92–94% correlated, so
averaging has little to fix. **One model is faster, 17× smaller, and easier to explain to a hotel manager**, so XGBoost stays.

The same holds more widely: XGBoost 0.7612, Random Forest 0.7548, Neural Network 0.7426 in cross-validation;
class weighting never helped; extra Kaggle features added +0.0026 together. **Every honest approach
lands at about 0.76–0.77 PR-AUC.** That ceiling comes from the data (what is known when the booking is made),
not from the choice of model. The only big "gains" we found came from leakage: post-booking columns (+0.0407)
and duplicate rows (0.9225).

## 6. Limitations

From [project_record.md](project_record.md) §10:

1. **The test set was looked at more than once.** Notebook 10 reads it a second time (after the decision was
   made on training data), so the test score is not a perfectly untouched estimate.
2. **93 test bookings have an identical profile in training.** A random split cannot keep look-alike bookings
   on one side.
3. **The threshold is slightly optimistic.** It was chosen on out-of-fold training predictions, from settings
   picked using all training rows.
4. **No time-based test.** We did not train on older bookings and test on newer ones; the arrival date is not
   the date the booking was made.
5. **Scope.** Two Portuguese hotels, 2015–2017, before 2020. The model is only claimed for bookings like these.
6. **Best-F1 is not best-for-business.** The threshold 0.335 maximises F1 (a balance of precision and recall),
   not money. The right threshold depends on real costs.
7. **Booking-time data cannot be fully proven.** The source dataset does not guarantee every kept column (e.g.
   `adr`, `deposit_type`) was never updated after the booking was made (project_record.md §2).

## 7. Next steps for the hotel

1. **Set the threshold from real costs.** Put a number on "cost of contacting a guest" and "value of saving a
   cancelled room", then pick the threshold that saves the most money instead of the F1-best one.
2. **Retrain on newer data**, ideally with the real booking-creation date, and test on the most recent months
   (time-based test) before trusting it.
3. **Collect a booking ID**, so real duplicates can be told apart from separate bookings.
4. **Start with a small trial**: use flags only for reminders for a few months, and measure whether
   cancellations actually fall.
5. **Monitor fairness**: check regularly that flags are not leading to different treatment by country.

The demo app (`streamlit run app/app.py`) shows how a flag would look for a single booking.
