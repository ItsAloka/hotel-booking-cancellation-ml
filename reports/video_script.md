# Demo video: recording guide and script

**Length:** 3 minutes · **Words to read:** about 420 · **Speak slowly and calmly.**

How to read this guide:

- 🖱️ **DO** = what to click or show on the screen.
- 🎤 **SAY** = read this out loud, word for word.

---

## Part A: Get ready (before you press record)

Do these once, in this order.

**1. Start the demo app.** Open a terminal in the project folder and run:

```
.venv\Scripts\streamlit run app/app.py
```

A browser tab opens at `http://localhost:8501`. Leave it open.
In the left sidebar, check that **"XGBoost (final model)"** is selected.

**2. Open these four things, each in its own window or tab:**

| # | What | Where to scroll to |
| --- | --- | --- |
| 1 | The final report (Word or Google Docs) | The cover page |
| 2 | `notebooks/01_eda_cleaning.ipynb` | Section **5. Duplicate rows** |
| 3 | `notebooks/08_model_selection.ipynb` | Section **1. The leaderboard** (the chart) |
| 4 | The demo app in the browser | The **"Predict one booking"** tab |

**3. Clean the screen.** Close other apps, turn off notifications (Windows: Focus / Do not disturb),
and zoom notebooks to about 125% so text is readable in the video.

**4. Start the screen recorder.**

- Easiest: press **Win + Alt + R** to start and stop recording (Xbox Game Bar). The video is saved in
  `Videos\Captures`.
- Or use **OBS Studio** (free) if you want to record the whole screen and your microphone.
- Test first: record 10 seconds, play it back, and check that your voice is clear.

**5. Practise once with a timer.** If you go over 3:00, shorten Scene 3 first.

---

## Part B: The script

### Scene 1: The problem (0:00 – 0:20)

🖱️ **DO:** Show the **report cover page**.

🎤 **SAY:**

> Hello, we are group 2026-AI-46, and this is our IT3091 Machine Learning project.
> We predict **hotel booking cancellations**.
> When a guest cancels, a room the hotel thought was sold can end up empty.
> If the hotel knows early which bookings are risky, it has time to act.

---

### Scene 2: The data and cleaning (0:20 – 0:55)

🖱️ **DO:** Switch to **notebook 01**, section **5. Duplicate rows**.

🎤 **SAY:**

> We used the Hotel Booking Demand dataset: two hotels in Portugal, from 2015 to 2017,
> with 119,390 bookings.
> First, we found 34,239 exact duplicate rows. Keeping them made the model look amazing,
> but it was only memorising copies, so we removed them.

🖱️ **DO:** Scroll **up** to section **4.1 A column that is too good to be true**.

🎤 **SAY:**

> We also found a trap. 7,409 bookings asked for a parking space, and **none** of them cancelled.
> That is because parking is recorded when the guest arrives, so it gives the answer away.
> We removed every column that is filled in after the booking is made.
> After cleaning, we had 84,969 bookings, and 27.7% of them were cancelled.

---

### Scene 3: Comparing the models (0:55 – 1:35)

🖱️ **DO:** Switch to **notebook 08**, section **1. The leaderboard**. Point the mouse at the bar chart.

🎤 **SAY:**

> Only 27.7% of bookings cancel. So a model that always says "not cancelled" is 72% accurate,
> but it never catches a single cancellation. That is why we did not use accuracy.
> We used **PR-AUC**, which focuses on the cancelled bookings.
>
> We trained a guessing baseline and five models: Logistic Regression, Decision Tree,
> Random Forest, XGBoost and a Neural Network.
> Every model used the same data, the same five cross-validation folds and the same score,
> so the comparison is fair.
> XGBoost was the best, at 0.761. That is more than 2.7 times better than guessing.

---

### Scene 4: The final test (1:35 – 2:05)

🖱️ **DO:** In **notebook 08**, scroll to section **3. Opening the test set**. Show the confusion matrix.

🎤 **SAY:**

> We kept 20% of the data as a test set and opened it only once, at the very end.
> On the test set, XGBoost scored a PR-AUC of 0.770.
> It catches **77% of cancellations**, and about 65% of the bookings it flags really do cancel.
> We also tested class weighting and an ensemble of three models.
> Neither gave a real improvement, so we kept the simpler model.

---

### Scene 5: The demo app (2:05 – 2:35)

🖱️ **DO:** Switch to the **app** in the browser. Click **"High-risk example"**, then the blue **Predict** button.
Wait for the result.

🎤 **SAY:**

> Here is our saved model in a small web app.
> This booking was made 250 days ahead, through an online travel agent, with no special requests.
> The model says the chance of cancelling is about 99.6%, so it is flagged as high risk.

🖱️ **DO:** Scroll up, click **"Low-risk example"**, then **Predict** again.

🎤 **SAY:**

> This one is a short-notice, direct booking from a returning guest with two special requests.
> The model gives it only about 0.5%, so it is not flagged.

---

### Scene 6: Recommendation and limits (2:35 – 2:52)

🖱️ **DO:** Switch to the **report, section 9** (Recommendation, risks and limitations).

🎤 **SAY:**

> Our advice to the hotel: score every new booking. For flagged bookings, send a friendly reminder,
> ask for a deposit, or plan overbooking. Never treat guests worse because of their country.
> The limits: the data comes from two Portuguese hotels before 2020,
> and the model misses some short-notice cancellations.

---

### Scene 7: Close (2:52 – 3:00)

🖱️ **DO:** Switch back to the **report cover page**.

🎤 **SAY:**

> All our code, notebooks and the full report are on GitHub. Thank you for watching.

🖱️ **DO:** Stop the recording (**Win + Alt + R** again).

---

## Part C: After recording

1. **Watch it once.** Check that the length is under 3:00 and your voice is clear.
   To cut the start or end, open it in the Windows **Photos** or **Clipchamp** app and use **Trim**.
2. **Upload to YouTube:** go to youtube.com → **Create → Upload video** → choose the file.
3. **Title:** `IT3091 ML Project – Hotel Booking Cancellation Prediction (Group 2026-AI-46)`
4. **Visibility:** choose **Unlisted**. Not Private, because then the examiner cannot watch it.
5. **Copy the link** and send it to Claude, who will put it in the report (cover page and section 10).
