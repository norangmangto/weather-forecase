# Data Ingestion Pipeline

This document explains how the system gathers weather and environmental data from various sources to prepare it for machine learning.

## 1. Data Sources

The pipeline ingests data from two primary providers:

### A. Deutscher Wetterdienst (DWD)
The DWD provides high-quality daily climate observations for Germany. We split this into two categories:
*   **Historical Archive**: Large ZIP files containing data from several decades ago up until the end of the previous year.
*   **Recent Data**: Smaller ZIP files containing data from the last few months up to a few days ago.

**Parameters collected**: Temperature (mean, max, min), ground temperature, precipitation, sunshine duration, snow depth, wind speed (mean, max), humidity, pressure, and cloud cover.

### B. Open-Meteo
Open-Meteo is used to enrich our dataset with environmental and supplemental weather data.
*   **Air Quality API**: Provides daily concentrations of PM10, PM2.5, Carbon Monoxide (CO), Nitrogen Dioxide (NO2), Sulphur Dioxide (SO2), and Ozone (O3).
*   **Historical Weather API (Supplemental)**: Provides UV Index and Visibility data, which are not always available in the standard DWD sets.

## 2. Multi-Station Support

The system is designed to handle multiple weather stations simultaneously. Each station is defined by:
*   `id`: The unique DWD station ID (e.g., `01078` for Dusseldorf).
*   `name`: A human-readable name.
*   `lat` / `lon`: Coordinates used for Open-Meteo API calls.

When data is ingested, a `station_id` column is added to every record to ensure that data from different cities remains isolated during feature engineering.

## 3. The Ingestion Process (`src/ingest.py`)

The `WeatherIngestor` class performs the following sequence for each station:

1.  **Download & Extract**: Fetches ZIP files from DWD Open Data servers and extracts the raw text files (usually named `produkt_klima_tag_*.txt`).
2.  **Clean & Transform**: Uses `pandas` to:
    *   Strip whitespace from column names.
    *   Handle missing values (DWD uses `-999.0` for nulls).
    *   Cast `station_id` to a padded 5-character string (e.g., `1078` -> `01078`).
3.  **Storage**: Data is saved into **DuckDB** tables:
    *   `raw_dwd_kl`: Climate observations.
    *   `raw_openmeteo_aq`: Air quality metrics.
    *   `raw_openmeteo_supplement`: UV and Visibility metrics.
4.  **Deduplication**: Since the Historical and Recent DWD files often overlap, a deduplication step runs at the end to ensure each date-station pair is unique.

## 4. How to Run

To refresh the raw data, run:
```bash
uv run src/ingest.py
```
This will clear existing tables and perform a fresh ingestion for all defined stations.
