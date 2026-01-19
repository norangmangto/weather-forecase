# Training and Prediction Pipeline

This document details how the system transforms raw data into features, trains machine learning models, and generates recursive forecasts.

## 1. Feature Engineering (dbt)

Raw data is processed using **dbt (data build tool)** to create the `weather_features` table. This layer is critical for providing context to the ML models.

### Key Transformations
*   **Station Partitioning**: All calculations are isolated per `station_id`.
*   **Temporal Features**: To capture seasonality, we transform `day_of_year` into `sine` and `cosine` waves. This tells the model that December 31st and January 1st are close together.
*   **Lag Features**: We include the values from 1 day and 7 days ago. This captures both short-term momentum and weekly cycles.
*   **Rolling Averages**: 7-day windows help the model understand the broader weather context (e.g., "Is it a hot week?" vs "Is it just a hot day?").

## 2. Model Training (`src/train.py`)

The system uses two powerful algorithms: **XGBoost** and **Random Forest**.

### Training Logic
*   **Per-Station Models**: Models are trained independently for each city. A model trained on Berlin data is never used to predict Dusseldorf weather.
*   **Multi-Target Training**: We train a separate regressor for every target (temp, wind, etc.) and a classifier for rain probability.
*   **Warm Starts**: To save time, the trainer attempts to load existing models. If the feature set is identical, it performs additional training (Boosting for XGBoost, more trees for Random Forest) rather than starting from scratch.
*   **Model Storage**: Models are saved in `models/` with the naming convention:
    `{algorithm}_{target}_{station_id}_{timestamp}.pkl`

## 3. Forecasting Logic (`src/predict.py`)

Since we only have access to historical data at prediction time, we use a **Recursive Forecasting** strategy to predict the next 7 days.

### Recursive Loop
1.  **Day 1**: Predict all weather variables using real historical data from "Yesterday".
2.  **Day 2**: Predict variables using the **predicted** values from Day 1 as if they were real observations.
3.  **Day 3-7**: Repeat the process, each time using the previous day's predictions to build the next day's features.

### Command Line Interface (CLI)
You can generate forecasts manually using the CLI:
```bash
# Forecast for Dusseldorf (01078)
uv run src/predict.py --station 01078

# Forecast for Berlin-Tegel (00433)
uv run src/predict.py --station 00433

# Forecast starting from a specific date
uv run src/predict.py --station 01975 --date 2024-01-01
```

## 4. Monitoring (`src/monitor.py`)
Predictions are automatically saved to the `historical_forecasts` table in DuckDB. This allows for future "backtesting" where you can compare the predictions made today against the actual weather recorded next week.
