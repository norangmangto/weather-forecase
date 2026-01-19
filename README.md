# Weather Prediction Application 🌦️

A complete, end-to-end machine learning pipeline for predicting 7-day weather forecasts.

## 🎯 Project Goal
To build a system that:
1.  **Ingests** historical weather data automatically.
2.  **Cleans and Transforms** the raw data into usable features.
3.  **Trains** machine learning models to understand weather patterns.
4.  **Predicts** future weather (Temperature, Rain, Wind, Humidity).

## 🏗️ Architecture & Tech Stack

We chose a modern, efficient stack to keep the project lightweight yet powerful.

### 1. **Environment Management: `uv`**
*   **What it is**: A super-fast replacement for `pip` and `virtualenv`.
*   **Why we use it**: It manages Python dependencies and virtual environments instantly, so you don't have to worry about "it works on my machine" issues.

### 2. **Database: `DuckDB`**
*   **What it is**: An in-process SQL OLAP database. Think of it as "SQLite for Analytics".
*   **Why we use it**: It handles large datasets (millions of rows) incredibly fast using a single file (`data/weather.db`). No need to set up a complex Postgres or MySQL server.

### 3. **Transformation: `dbt` (Data Build Tool)**
*   **What it is**: The industry standard for data transformation. You write SQL, and dbt handles the dependency graphs and table creation.
*   **Why we use it**: It brings software engineering best practices (version control, testing, modularity) to SQL.
    *   **Staging Models**: Clean up raw variable names (e.g., renaming `TMK` to `temp_mean`).
    *   **Marts**: Combine data and calculate features like 7-day rolling averages.

### 4. **Machine Learning: `XGBoost` & `Random Forest`**
*   **What they are**: Ensemble learning methods that combine many "weak" decision trees to make a strong prediction.
*   **Why we use them**:
    *   **Tabular Data**: These are currently the best algorithms for structured data like weather records.
    *   **Robustness**: Random Forest is great at handling noise/outliers.
    *   **Performance**: XGBoost is highly optimized and often achieves the highest accuracy.

## 📂 Project Structure Explained

```
weather-forecast/
├── data/
│   └── weather.db              # The DuckDB database file (created after running ingestion)
├── dbt/                        # dbt project folder
│   ├── models/
│   │   ├── staging/            # Initial cleaning of raw data
│   │   └── marts/              # Final tables ready for ML (feature engineering)
│   └── dbt_project.yml         # dbt configuration
├── models/                     # Saved trained models (.pkl files)
├── src/
│   ├── ingest.py               # Script to download and save raw data
│   ├── train.py                # Script to load data, train models, and save them
│   └── predict.py              # Script to load models and predict the next 7 days
├── pyproject.toml              # Project dependencies (managed by uv)
└── README.md                   # You are here!
```

## 🚀 Quick Start Guide

Follow these steps to run the pipeline from scratch.

### Step 1: Ingest Data
Download historical data from DWD (German Weather Service) and Open-Meteo.
```bash
uv run src/ingest.py
```
*   *What happens*: Connects to DWD FTP, downloads a ZIP file, extracts it, and saves ~26k rows to DuckDB.

### Step 2: Transform Data
Run dbt to clean and prepare the data.
```bash
cd dbt && uv run dbt run --profiles-dir .
```
*   *What happens*: dbt reads the raw data, applies SQL transformations (renaming, averages, lags), and saves clean tables back to DuckDB.

### Step 3: Train Models
Train the Machine Learning models.
```bash
uv run src/train.py
```
*   *What happens*: Loads features from DuckDB, splits data into training/testing sets, trains 14 models (2 algorithms x 7 targets), and saves them to `models/`.

### Step 4: Predict
Generate the forecast.
```bash
uv run src/predict.py
```
*   *What happens*: Loads the saved models, calculates the features for "tomorrow", and predicts weather for the next 7 days. Check the generated `.csv` files for results!
