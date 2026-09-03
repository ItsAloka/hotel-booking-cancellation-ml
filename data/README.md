# Dataset record

## Hotel Booking Demand

- **File:** `raw/hotel_bookings.csv`
- **Source:** https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv
- **Downloaded:** 2026-09-03
- **Rows:** 119,390 data rows (plus header)
- **SHA-256:** `7C2AE42A7353905EA136E5C2287F17C92C5435826598BFBB8491C6F0C7B1FC06`
- **Provenance:** TidyTuesday's published copy of the Hotel Booking Demand data, originally described by Antonio, Almeida and Nunes (2019), *Hotel booking demand datasets*, Data in Brief 22, DOI: 10.1016/j.dib.2018.11.126.

The CSV includes the expected `is_canceled` target and 32 columns. The analysis must exclude `reservation_status` and `reservation_status_date` from model features because they reveal the outcome.

The raw file is retained unchanged. Any derived data must be written separately under `data/processed/` and generated reproducibly.
