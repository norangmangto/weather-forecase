import duckdb
import pandas as pd
import numpy as np
import joblib
import os
from glob import glob
from datetime import datetime, timedelta
from src.monitor import WeatherMonitor

class WeatherPredictor:
    def __init__(self, db_path="data/weather.db", models_dir="models"):
        self.db_path = db_path
        self.models_dir = models_dir
        self.models = {}
        self.monitor = WeatherMonitor(db_path)

    def load_latest_models(self):
        """Load the most recent model for each target."""
        model_files = glob(os.path.join(self.models_dir, "*.pkl"))

        # Group by target and model_type
        models_dict = {}
        for model_file in model_files:
            basename = os.path.basename(model_file)
            parts = basename.replace('.pkl', '').split('_')

            # Parse filename: modeltype_target_timestamp.pkl
            # Classification models might have different naming pattern if not careful,
            # but based on train.py they follow {model_name}_{target_name}_{timestamp}.pkl
            # model_name is 'xgboost' or 'random_forest'

            model_type = parts[0]  # xgboost or random
            if model_type == "random":
                model_type = "_".join(parts[:2])  # random_forest
                target = "_".join(parts[2:-2])
                timestamp = "_".join(parts[-2:])
            else:
                target = "_".join(parts[1:-2])
                timestamp = "_".join(parts[-2:])

            key = f"{model_type}_{target}"

            if key not in models_dict or timestamp > models_dict[key]['timestamp']:
                models_dict[key] = {
                    'path': model_file,
                    'timestamp': timestamp,
                    'model_type': model_type,
                    'target': target
                }

        # Load the models
        for key, info in models_dict.items():
            self.models[key] = {
                'model': joblib.load(info['path']),
                'target': info['target'],
                'model_type': info['model_type']
            }
            print(f"Loaded {key} from {info['path']}")

    def get_features_for_date(self, target_date=None):
        """
        Get features to predict FOR target_date.
        This means fetching the row from target_date - 1 day.
        If target_date is None, fetch the absolute latest row.
        """
        conn = duckdb.connect(self.db_path)

        if target_date:
            # We need data from the day BEFORE the target start date
            ref_date = pd.to_datetime(target_date) - timedelta(days=1)
            ref_date_str = ref_date.strftime('%Y-%m-%d')
            query = f"SELECT * FROM weather_features WHERE measurement_date = '{ref_date_str}'"
            print(f"Fetching features for reference date: {ref_date_str} (to predict starting {target_date})")
        else:
            query = """
                SELECT * FROM weather_features
                WHERE measurement_date IS NOT NULL
                ORDER BY measurement_date DESC
                LIMIT 1
            """
            print("Fetching latest available features...")

        df = conn.execute(query).df()
        conn.close()

        if df.empty:
            raise ValueError(f"No data found for reference date. Cannot predict for {target_date}.")

        # Check for data freshness
        ref_date = pd.to_datetime(df['measurement_date'].iloc[0])
        today = pd.to_datetime(datetime.now().date())
        days_lag = (today - ref_date).days

        if days_lag > 30 and not target_date:
            print(f"\n========================================================")
            print(f"⚠️  WARNING: Using historical data from {ref_date.strftime('%Y-%m-%d')}!")
            print(f"   The database does not contain recent weather data.")
            print(f"   Forecast will be for the period starting: {(ref_date + timedelta(days=1)).strftime('%Y-%m-%d')}")
            print(f"========================================================\n")

        return df

    def prepare_forecast_input(self, latest_row, day_offset=0):
        """Prepare input features for prediction."""
        # Calculate future date
        base_date = pd.to_datetime(latest_row['measurement_date'].iloc[0])
        future_date = base_date + timedelta(days=day_offset + 1)

        # Create feature vector
        features = {
            'day_of_year': future_date.dayofyear,
            'day_sin': np.sin(2 * np.pi * future_date.dayofyear / 365.0),
            'day_cos': np.cos(2 * np.pi * future_date.dayofyear / 365.0),
            'month': future_date.month,
            'year': future_date.year,
            'temp_mean_lag1': latest_row['temp_mean'].iloc[0],
            'temp_mean_lag7': latest_row['temp_mean_lag7'].iloc[0],
            'temp_max_lag1': latest_row['temp_max'].iloc[0],
            'temp_min_lag1': latest_row['temp_min'].iloc[0],
            'temp_ground_min_lag1': latest_row['temp_ground_min'].iloc[0] if 'temp_ground_min' in latest_row else 0,
            'wind_speed_lag1': latest_row['wind_speed_mean'].iloc[0],
            'wind_gust_lag1': latest_row['wind_gust_max'].iloc[0] if 'wind_gust_max' in latest_row else 0,
            'precipitation_lag1': latest_row['precipitation_mm'].iloc[0],
            'humidity_lag1': latest_row['humidity_mean'].iloc[0],
            'pressure_lag1': latest_row['pressure_hpa'].iloc[0] if 'pressure_hpa' in latest_row else 0,
            'snow_depth_lag1': latest_row['snow_depth_cm'].iloc[0] if 'snow_depth_cm' in latest_row else 0,
            'uv_index_lag1': latest_row['uv_index_max'].iloc[0] if 'uv_index_max' in latest_row else 0,
            'temp_mean_7d_avg': latest_row['temp_mean_7d_avg'].iloc[0],
            'temp_max_7d_avg': latest_row['temp_max_7d_avg'].iloc[0],
            'temp_min_7d_avg': latest_row['temp_min_7d_avg'].iloc[0],
            'wind_speed_7d_avg': latest_row['wind_speed_7d_avg'].iloc[0],
            'wind_gust_7d_avg': latest_row['wind_gust_7d_avg'].iloc[0] if 'wind_gust_7d_avg' in latest_row else 0,
            'precipitation_7d_avg': latest_row['precipitation_7d_avg'].iloc[0],
            'humidity_7d_avg': latest_row['humidity_7d_avg'].iloc[0],
            'pressure_7d_avg': latest_row['pressure_7d_avg'].iloc[0] if 'pressure_7d_avg' in latest_row else 0,
            'pm10_mean': latest_row['pm10_mean'].iloc[0],
            'pm2_5_mean': latest_row['pm2_5_mean'].iloc[0],
            'carbon_monoxide_mean': latest_row['carbon_monoxide_mean'].iloc[0] if 'carbon_monoxide_mean' in latest_row else 0,
            'nitrogen_dioxide_mean': latest_row['nitrogen_dioxide_mean'].iloc[0] if 'nitrogen_dioxide_mean' in latest_row else 0,
            'sulphur_dioxide_mean': latest_row['sulphur_dioxide_mean'].iloc[0] if 'sulphur_dioxide_mean' in latest_row else 0,
            'ozone_mean': latest_row['ozone_mean'].iloc[0] if 'ozone_mean' in latest_row else 0,
            'uv_index_max': latest_row['uv_index_max'].iloc[0] if 'uv_index_max' in latest_row else 0,
            'visibility_mean': latest_row['visibility_mean'].iloc[0] if 'visibility_mean' in latest_row else 0,
            'cloud_cover_mean': latest_row['cloud_cover_mean'].iloc[0],
            'sunshine_hours': latest_row['sunshine_hours'].iloc[0]
        }

        return pd.DataFrame([features])

    def predict_7_days(self, model_type='xgboost', start_date=None):
        """Generate 7-day forecast for all weather variables."""
        try:
            latest_row = self.get_features_for_date(start_date)
        except ValueError as e:
            print(f"Error: {e}")
            return pd.DataFrame()

        base_date = pd.to_datetime(latest_row['measurement_date'].iloc[0])

        print(f"\nGenerating 7-day forecast using {model_type} models...")
        # The base_date is the reference date (T-1). The first prediction is for base_date + 1 day.
        print(f"Reference data date: {base_date.strftime('%Y-%m-%d')}")
        print(f"First prediction date: {(base_date + timedelta(days=1)).strftime('%Y-%m-%d')}\n")

        forecasts = []

        for day in range(7):
            forecast_date = base_date + timedelta(days=day + 1)
            features = self.prepare_forecast_input(latest_row, day)

            predictions = {'date': forecast_date.strftime('%Y-%m-%d')}

            # Predict each target
            targets = [
                'temp_max', 'temp_min', 'temp_mean', 'temp_ground_min',
                'wind_speed_mean', 'wind_gust_max', 'precipitation_mm',
                'humidity_mean', 'pressure_hpa', 'snow_depth_cm',
                'uv_index_max', 'visibility_mean', 'rain_probability'
            ]
            for target in targets:
                model_key = f"{model_type}_{target}"
                if model_key in self.models:
                    model = self.models[model_key]['model']
                    if target == 'rain_probability':
                        # For probability tasks, use predict_proba
                        # Check if model has predict_proba
                        if hasattr(model, 'predict_proba'):
                            pred = model.predict_proba(features)[0][1] # Probability of class 1 (Rain)
                        else:
                            pred = model.predict(features)[0] # Fallback if regressor used by mistake
                    else:
                        pred = model.predict(features)[0]
                    predictions[target] = pred
                else:
                    predictions[target] = None

            forecasts.append(predictions)

        return pd.DataFrame(forecasts)

    def run(self, start_date=None):
        """Main prediction pipeline."""
        print("Loading trained models...")
        self.load_latest_models()

        print(f"\nAvailable models: {list(self.models.keys())}")

        # Generate forecasts for both model types
        xgb_forecast = self.predict_7_days('xgboost', start_date)
        rf_forecast = self.predict_7_days('random_forest', start_date)

        if xgb_forecast.empty or rf_forecast.empty:
            print("Prediction failed due to missing data.")
            return

        print("\n" + "="*100)
        print("7-DAY WEATHER FORECAST - XGBoost")
        print("="*100)
        print(xgb_forecast.to_string(index=False))

        print("\n" + "="*100)
        print("7-DAY WEATHER FORECAST - Random Forest")
        print("="*100)
        print(rf_forecast.to_string(index=False))

        # Save forecasts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_suffix = f"_{start_date}" if start_date else ""
        xgb_forecast.to_csv(f"forecasts_xgboost{date_suffix}_{timestamp}.csv", index=False)
        rf_forecast.to_csv(f"forecasts_rf{date_suffix}_{timestamp}.csv", index=False)

        # Log to MLOps Monitor
        self.monitor.save_forecast(xgb_forecast, 'xgboost')
        self.monitor.save_forecast(rf_forecast, 'random_forest')

        print(f"\nForecasts saved to:")
        print(f"  - forecasts_xgboost{date_suffix}_{timestamp}.csv")
        print(f"  - forecasts_rf{date_suffix}_{timestamp}.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate 7-day weather forecast.')
    parser.add_argument('--date', type=str, help='Start date for prediction (YYYY-MM-DD). Defaults to tomorrow relative to latest data.')
    args = parser.parse_args()

    predictor = WeatherPredictor()
    predictor.run(start_date=args.date)
