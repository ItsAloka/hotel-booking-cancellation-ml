# Dataset source and fingerprint

| Field | Value |
| --- | --- |
| Dataset | Hotel Booking Demand |
| File | `data/raw/hotel_bookings.csv` |
| Source URL | https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv |
| Mirror of | TidyTuesday 2020-02-11 "Hotel bookings" (originally cleaned by Thomas Mock & Antoine Bichat) |
| Download date | 2026-09-03 |
| Size (bytes) | 16,855,599 |
| SHA-256 | `7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06` |
| Rows (excl. header) | 119,390 |
| Columns | 32 |
| Target column | `is_canceled` (values {0, 1}; overall cancellation rate 37.04%) |
| Hotels | Resort Hotel, City Hotel |
| Arrival years | 2015, 2016, 2017 |
| Licence | CC0 1.0 (public domain) |

## Academic citation

Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets.*
Data in Brief, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126

## Verifying the fingerprint

```bash
sha256sum data/raw/hotel_bookings.csv
# expected: 7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06
```

## Leakage note

`reservation_status` (Check-Out / Canceled / No-Show) and `reservation_status_date`
are recorded at or after the booking outcome and MUST NOT be used as model features.
See `knowledge/` vault note "Hotel Booking Demand: reservation_status and
reservation_status_date leak the target".
