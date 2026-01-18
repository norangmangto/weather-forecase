import duckdb
import pandas as pd
from datetime import datetime

class WeatherMonitor:
    def __init__(self, db_path="data/weather.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the historical forecasts table."""
        conn = duckdb.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS historical_forecasts (
                prediction_time TIMESTAMP,
                target_date DATE,
                model_type VARCHAR,
                target_name VARCHAR,
                predicted_value DOUBLE,
                prediction_horizon_days INTEGER
            )
        """)
        conn.close()

    def save_forecast(self, forecast_df, model_type):
        """
        Saves a 7-day forecast dataframe to the historical tracking table.
        df columns: [date, temp_max, temp_min, ..., rain_probability]
        """
        conn = duckdb.connect(self.db_path)
        prediction_time = datetime.now()

        # Melt dataframe to long format for easier storage and comparison
        # Base columns are 'date' (which is the target_date)
        melted = forecast_df.melt(id_vars=['date'], var_name='target_name', value_name='predicted_value')
        melted['prediction_time'] = prediction_time
        melted['model_type'] = model_type
        melted['target_date'] = pd.to_datetime(melted['date']).dt.date

        # Calculate horizon (days between prediction and target)
        base_date = pd.to_datetime(forecast_df['date'].iloc[0]) - pd.Timedelta(days=1)
        melted['prediction_horizon_days'] = (pd.to_datetime(melted['date']) - base_date).dt.days

        # Insert into DuckDB
        # We use a temp table for the append to avoid complex SQL generation for many rows
        conn.execute("CREATE TEMPORARY TABLE temp_forecast AS SELECT * FROM melted")
        conn.execute("INSERT INTO historical_forecasts SELECT prediction_time, target_date, model_type, target_name, predicted_value, prediction_horizon_days FROM temp_forecast")

        conn.close()
        print(f"Saved {len(melted)} prediction points to historical_forecasts for {model_type}")

    def get_accuracy_metrics(self):
        """
        Compares historical forecasts with actual values from weather_features table.
        Returns a dataframe with MAE per target and model.
        """
        conn = duckdb.connect(self.db_path)

        # Join historical_forecasts (predictions) with weather_features (actuals)
        # Note: target_name in historical_forecasts might need mapping to actual column names
        # if they differ. Here they match our train/predict logic.

        # We first need to get a list of target names to compare
        targets = conn.execute("SELECT DISTINCT target_name FROM historical_forecasts").df()['target_name'].tolist()

        all_metrics = []

        for target in targets:
            if target == 'date': continue

            # Special case for rain_probability vs rain_prob (check column name in features)
            actual_col = target
            if target == 'rain_probability':
                # Map rain_probability (pred) to rain_prob (actual classification target)
                # Actually, weather_features has 'precipitation_mm'. Let's use that for regression.
                # For classification, we might need to derive 'was_rainy' if we don't have it.
                continue

            query = f"""
                SELECT
                    f.model_type,
                    f.target_name,
                    AVG(ABS(f.predicted_value - a.{actual_col})) as mae,
                    SQRT(AVG(POWER(f.predicted_value - a.{actual_col}, 2))) as rmse,
                    COUNT(*) as sample_size
                FROM historical_forecasts f
                JOIN weather_features a ON f.target_date = a.measurement_date
                WHERE f.target_name = '{target}'
                AND a.{actual_col} IS NOT NULL
                GROUP BY 1, 2
            """
            try:
                metrics = conn.execute(query).df()
                if not metrics.empty:
                    all_metrics.append(metrics)
            except Exception as e:
                # Column might not exist in actuals or other SQL error
                continue

        conn.close()

        if not all_metrics:
            return pd.DataFrame()

        return pd.concat(all_metrics).reset_index(drop=True)

if __name__ == "__main__":
    monitor = WeatherMonitor()
    print("Historical Performance Metrics:")
    metrics = monitor.get_accuracy_metrics()
    if metrics.empty:
        print("No historical data available for evaluation yet. Start by generating forecasts!")
    else:
        print(metrics.to_string(index=False))
