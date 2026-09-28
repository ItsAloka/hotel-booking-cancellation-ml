# Problem framing canvas

**Group 2026-AI-46 · Guided Data Track, code 6: Tourism & Hospitality (Hotel Booking Demand).**
Numbers come from [project_record.md](project_record.md) and the notebooks named in brackets.

| Box | Our answer |
| --- | --- |
| **Business scenario** | A hotel chain wants to improve booking reliability, revenue planning and customer management (assignment brief, code 6). |
| **Stakeholder** | The **hotel revenue manager**: the person who decides how many rooms to sell, when to overbook, and which guests to contact. |
| **Decision need** | **Which new bookings should we follow up?** For example: send a friendly reminder, ask for a deposit, or count the booking as "at risk" when planning overbooking. |
| **Primary lens** | **Cancellation risk.** |
| **Secondary lens** | **None.** Segmentation or operational planning would be separate projects; adding them would not make the cancellation decision better, and the brief says extra work without a clear link earns no marks. |
| **Why this lens** | After cleaning, **27.7%** of bookings are cancelled (notebook 01 §5). An empty room that was "sold" is lost revenue. If the hotel knows *which* bookings are risky early, it has time to act. |
| **Unit of analysis** | **One booking**, scored **at the moment it is made**. One row of the data = one booking. |
| **ML task** | **Binary classification**: cancelled (1) or not cancelled (0). Target column: `is_canceled`. |
| **Output** | (1) A **cancellation probability** between 0 and 1, and (2) a **yes/no flag**: "high risk" when the probability is **0.335 or more** (threshold chosen on training data, notebook 06). |
| **Success metric** | **PR-AUC** (precision–recall area; how well the model ranks cancelled bookings above the rest) as the main score, with **recall** (share of real cancellations we catch) next to it. |
| **Why not accuracy** | A "model" that always says *not cancelled* is already **72.25% accurate** and catches zero cancellations (notebook 08 §1). Accuracy hides this; PR-AUC does not. The no-skill PR-AUC is **0.2775**, so every model is measured against that floor. |
| **Constraint: inputs** | **Only information known when the booking is made.** Columns filled in later (final status, parking, assigned room, booking changes, waiting list) are removed, even though they would raise the score by +0.0407 PR-AUC (notebook 01 §4). |
| **Constraint: data** | Two hotels in Portugal (one City, one Resort), arrivals July 2015 – August 2017, 119,390 raw rows → 84,969 bookings after cleaning. The model is only claimed to work for bookings like these. |
| **Constraint: people** | The group must be able to explain every step simply, so simpler models and methods win when scores are equal. |
| **What "done" looks like** | A saved model that scores a new booking, a test-set result the hotel can understand ("catches about 3 in 4 cancellations"), and a recommendation with its limits stated. |

## Result in one line

The final model, **XGBoost**, reaches **PR-AUC 0.770** on the test set and **catches 77% of cancellations**;
about **65% of the bookings it flags really cancel** (notebook 08). See [recommendation.md](recommendation.md).
