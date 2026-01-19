import duckdb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, roc_auc_score
from xgboost import XGBRegressor, XGBClassifier
import joblib
import os
from datetime import datetime

class WeatherModelTrainer:
    def __init__(self, db_path="data/weather.db", models_dir="models"):
        self.db_path = db_path
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

    def load_features(self):
        """Load the feature-engineered data from DuckDB."""
        conn = duckdb.connect(self.db_path)
        df = conn.execute("SELECT * FROM weather_features").df()
        conn.close()

        # Drop rows with NaN values (from lag/rolling operations)
        df = df.dropna()
        print(f"Loaded {len(df)} rows with complete features")

        return df

    def prepare_data(self, df, target_cols):
        """Prepare feature matrix and target variables."""
        # Define feature columns (exclude targets and metadata)
        feature_cols = [
            'day_of_year', 'day_sin', 'day_cos', 'month', 'year',
            'temp_mean_lag1', 'temp_mean_lag7',
            'temp_max_lag1', 'temp_min_lag1', 'temp_ground_min_lag1',
            'wind_speed_lag1', 'wind_gust_lag1', 'precipitation_lag1',
            'humidity_lag1', 'pressure_lag1', 'snow_depth_lag1', 'uv_index_lag1',
            'temp_mean_7d_avg', 'temp_max_7d_avg', 'temp_min_7d_avg',
            'wind_speed_7d_avg', 'wind_gust_7d_avg', 'precipitation_7d_avg',
            'humidity_7d_avg', 'pressure_7d_avg',
            'pm10_mean', 'pm2_5_mean', 'carbon_monoxide_mean', 'nitrogen_dioxide_mean',
            'sulphur_dioxide_mean', 'ozone_mean',
            'uv_index_max', 'visibility_mean',
            'cloud_cover_mean', 'sunshine_hours'
        ]

        # Filter only the columns that exist
        available_features = [col for col in feature_cols if col in df.columns]
        print(f"Using {len(available_features)} features: {available_features}")

        X = df[available_features]
        y = df[target_cols]

        return X, y

    def load_latest_model(self, model_type, target_name, station_id, current_features=None):
        """Load the most recent model if it exists and matches features."""
        from glob import glob

        # Pattern: modeltype_targetname_stationid_timestamp.pkl
        pattern = os.path.join(self.models_dir, f"{model_type}_{target_name}_{station_id}_*.pkl")
        files = glob(pattern)

        if not files:
            return None

        # Sort by timestamp (files[-1] is latest)
        latest_file = sorted(files)[-1]
        model = joblib.load(latest_file)

        # Validate features if current_features is provided
        if current_features is not None:
            model_features = []
            if hasattr(model, 'feature_names_in_'):
                model_features = list(model.feature_names_in_)
            elif hasattr(model, 'feature_names'):
                model_features = model.feature_names # XGBoost

            if model_features and list(model_features) != list(current_features):
                print(f"Feature mismatch for {latest_file}. Returning None for fresh training.")
                return None

        print(f"Loading existing {model_type} model for {target_name} ({station_id}) from {latest_file}")
        return model

    def train_regressors(self, X, y, target_name, station_id):
        """Train XGBoost and Random Forest regression models."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        models = {}
        results = {}

        # XGBoost
        print(f"\nTraining XGBoost Regressor for {target_name} (Station: {station_id})...")
        existing_xgb = self.load_latest_model('xgboost', target_name, station_id, current_features=X_train.columns)

        xgb_model = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0
        )

        if existing_xgb:
             # Use the existing model as a base for further training
            xgb_model.fit(X_train, y_train, xgb_model=existing_xgb)
        else:
            xgb_model.fit(X_train, y_train)

        y_pred = xgb_model.predict(X_test)

        models['xgboost'] = xgb_model
        results['xgboost'] = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

        # Random Forest
        print(f"Training Random Forest Regressor for {target_name} (Station: {station_id})...")
        existing_rf = self.load_latest_model('random_forest', target_name, station_id, current_features=X_train.columns)

        if existing_rf:
            # Warm start requires increasing n_estimators
            rf_model = existing_rf
            rf_model.set_params(n_estimators=rf_model.n_estimators + 50, warm_start=True)
            rf_model.fit(X_train, y_train)
        else:
            rf_model = RandomForestRegressor(
                n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
            )
            rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_test)

        models['random_forest'] = rf_model
        results['random_forest'] = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        return models, results

    def train_classifiers(self, X, y, target_name, station_id):
        """Train XGBoost and Random Forest classification models."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        models = {}
        results = {}

        # XGBoost Classifier
        print(f"\nTraining XGBoost Classifier for {target_name} (Station: {station_id})...")
        existing_xgb = self.load_latest_model('xgboost', target_name, station_id, current_features=X_train.columns)

        xgb_model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0,
            eval_metric='logloss'
        )

        if existing_xgb:
            xgb_model.fit(X_train, y_train, xgb_model=existing_xgb)
        else:
            xgb_model.fit(X_train, y_train)

        y_pred = xgb_model.predict(X_test)
        y_prob = xgb_model.predict_proba(X_test)[:, 1]

        models['xgboost'] = xgb_model
        results['xgboost'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }

        # Random Forest Classifier
        print(f"Training Random Forest Classifier for {target_name} (Station: {station_id})...")
        existing_rf = self.load_latest_model('random_forest', target_name, station_id, current_features=X_train.columns)

        if existing_rf:
            rf_model = existing_rf
            rf_model.set_params(n_estimators=rf_model.n_estimators + 50, warm_start=True)
            rf_model.fit(X_train, y_train)
        else:
            rf_model = RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
            )
            rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_test)
        y_prob = rf_model.predict_proba(X_test)[:, 1]

        models['random_forest'] = rf_model
        results['random_forest'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        return models, results

    def save_models(self, models, target_name, station_id):
        """Save trained models to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for model_name, model in models.items():
            model_path = os.path.join(self.models_dir, f"{model_name}_{target_name}_{station_id}_{timestamp}.pkl")
            joblib.dump(model, model_path)
            print(f"Saved {model_name} to {model_path}")

    def cleanup_old_models(self, keep_n=3):
        """Keep only the N most recent models for each target/type/station."""
        from glob import glob
        from collections import defaultdict

        all_model_files = glob(os.path.join(self.models_dir, "*.pkl"))

        # Group by type, target, and station
        groups = defaultdict(list)
        for f in all_model_files:
            basename = os.path.basename(f)
            parts = basename.replace('.pkl', '').split('_')
            # Extract prefix (type+target+station) and timestamp
            # We need to be careful with the varying lengths of target names
            # But we know timestamp is the last two parts: YYYYMMDD and HHMMSS
            prefix = "_".join(parts[:-2])
            groups[prefix].append(f)

        print("\nCleaning up old model checkpoints...")
        for prefix, files in groups.items():
            files = sorted(files)
            if len(files) > keep_n:
                to_delete = files[:-keep_n]
                print(f"  Removing {len(to_delete)} old versions of {prefix}")
                for f in to_delete:
                    try:
                        os.remove(f)
                    except Exception as e:
                        print(f"  Error deleting {f}: {e}")

    def run(self):
        """Main training pipeline."""
        print("Loading features from DuckDB...")
        full_df = self.load_features()

        stations = full_df['station_id'].unique()
        print(f"Found {len(stations)} stations: {stations}")

        regression_targets = {
            'temp_max': ['temp_max'],
            'temp_min': ['temp_min'],
            'temp_mean': ['temp_mean'],
            'temp_ground_min': ['temp_ground_min'],
            'wind_speed_mean': ['wind_speed_mean'],
            'wind_gust_max': ['wind_gust_max'],
            'precipitation_mm': ['precipitation_mm'],
            'humidity_mean': ['humidity_mean'],
            'pressure_hpa': ['pressure_hpa'],
            'snow_depth_cm': ['snow_depth_cm'],
            'uv_index_max': ['uv_index_max'],
            'visibility_mean': ['visibility_mean']
        }

        all_results = {}

        for station_id in stations:
            print(f"\n{'#'*60}")
            print(f"### TRAINING FOR STATION: {station_id}")
            print(f"{'#'*60}")

            df = full_df[full_df['station_id'] == station_id].copy()

            for target_name, target_cols in regression_targets.items():
                print(f"\n{'='*60}")
                print(f"Training REGRESSORS for: {target_name} ({station_id})")
                print(f"{'='*60}")

                X, y = self.prepare_data(df, target_cols)
                if y.shape[1] == 0:
                    print(f"Skipping {target_name} (column not found)")
                    continue

                models, results = self.train_regressors(X, y[target_cols[0]], target_name, station_id)
                all_results[f"{station_id}_{target_name}"] = results
                self.save_models(models, target_name, station_id)

                print(f"\n{target_name} Results ({station_id}):")
                for model_name, metrics in results.items():
                    print(f"  {model_name}: RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")

            # 2. Classification Target (Rain Probability)
            print(f"\n{'='*60}")
            print(f"Training CLASSIFIERS for: rain_probability ({station_id})")
            print(f"{'='*60}")

            df['is_raining'] = (df['precipitation_mm'] > 0.1).astype(int)
            X, y = self.prepare_data(df, ['is_raining'])
            models, results = self.train_classifiers(X, y['is_raining'], 'rain_probability', station_id)

            all_results[f"{station_id}_rain_probability"] = results
            self.save_models(models, 'rain_probability', station_id)

            print(f"\nrain_probability Results ({station_id}):")
            for model_name, metrics in results.items():
                print(f"  {model_name}: Accuracy={metrics['accuracy']:.4f}, AUC={metrics['roc_auc']:.4f}")

        # Cleanup old models
        self.cleanup_old_models(keep_n=3)

        print(f"\n{'='*60}")
        print("Training completed for all stations!")
        print(f"{'='*60}")

        return all_results

if __name__ == "__main__":
    trainer = WeatherModelTrainer()
    results = trainer.run()

if __name__ == "__main__":
    trainer = WeatherModelTrainer()
    results = trainer.run()
