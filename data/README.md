# Data layout

`raw/hotel_bookings.csv` is the unchanged 119,390-row Hotel Booking Demand file used in the project. See [source record](raw/SOURCE.md) for the Kaggle link, actual download URL, fingerprint and provenance limitations.

`processed/` is created by the notebooks and is not stored in Git (rebuild it by running them):

- `hotel_bookings_clean.csv` — notebook 01: cleaned data, duplicates removed, 84,969 bookings.
- `train.csv` (67,975 rows) and `test.csv` (16,994 rows) — notebook 02: the 80/20 split used by every model notebook.
