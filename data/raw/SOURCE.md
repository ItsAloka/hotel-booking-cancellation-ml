# Dataset source and exact local version

| Field | Value |
| --- | --- |
| Dataset | Hotel Booking Demand |
| Kaggle listing | https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand |
| Actual downloaded source | https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv |
| Local file | data/raw/hotel_bookings.csv |
| Recorded original download | 2026-09-03 |
| Provenance reviewed | 2026-09-17 |
| Size | 16,855,599 bytes |
| Shape | 119,390 rows; 32 columns |
| SHA-256 | 7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06 |

Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets.* Data in Brief, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126

The TidyTuesday dictionary/combining script is at https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-02-11 . It combines the study's two hotel tables and adds the hotel category. Kaggle was located as the requested dataset listing; the local bytes were not downloaded from or compared against Kaggle during this revision.

## Licence and access

The original article identifies CC BY 4.0: https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/ . This is the article's licence; it should not automatically be substituted for every mirror's dataset licence. The earlier local record asserted CC0, but the exact current Kaggle/mirror data licence was not independently verified in this revision. Preserve attribution and review the download provider's terms before redistribution. The raw CSV remains outside version control.

## Timing and leakage

Source extraction used change logs relative to pre-arrival time where available; it does not verify original booking-time snapshots. Outcome/status fields never enter model inputs; see reports/project_record.md §2 for which columns were excluded and why. The raw file remains unchanged; all derived files are under data/processed/.
