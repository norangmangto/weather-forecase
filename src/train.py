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
            'temp_max_lag1', 'temp_min_lag1',
            'wind_speed_lag1', 'precipitation_lag1', 'humidity_lag1',
            'temp_mean_7d_avg', 'temp_max_7d_avg', 'temp_min_7d_avg',
            'wind_speed_7d_avg', 'precipitation_7d_avg', 'humidity_7d_avg',
            'pm10_mean', 'pm2_5_mean', 'cloud_cover_mean', 'sunshine_hours'
        ]

        # Filter only the columns that exist
        available_features = [col for col in feature_cols if col in df.columns]
        print(f"Using {len(available_features)} features: {available_features}")

        X = df[available_features]
        y = df[target_cols]

        return X, y

    def load_latest_model(self, model_type, target_name):
        """Load the most recent model if it exists."""
        from glob import glob

        # Pattern: modeltype_targetname_timestamp.pkl
        pattern = os.path.join(self.models_dir, f"{model_type}_{target_name}_*.pkl")
        files = glob(pattern)

        if not files:
            return None

        # Sort by timestamp (files[-1] is latest)
        latest_file = sorted(files)[-1]
        print(f"Loading existing {model_type} model for {target_name} from {latest_file}")
        return joblib.load(latest_file)

    def train_regressors(self, X, y, target_name):
        """Train XGBoost and Random Forest regression models."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        models = {}
        results = {}

        # XGBoost
        print(f"\nTraining XGBoost Regressor for {target_name}...")
        existing_xgb = self.load_latest_model('xgboost', target_name)

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
        print(f"Training Random Forest Regressor for {target_name}...")
        existing_rf = self.load_latest_model('random_forest', target_name)

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

    def train_classifiers(self, X, y, target_name):
        """Train XGBoost and Random Forest classification models."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        models = {}
        results = {}

        # XGBoost Classifier
        print(f"\nTraining XGBoost Classifier for {target_name}...")
        existing_xgb = self.load_latest_model('xgboost', target_name)

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
        print(f"Training Random Forest Classifier for {target_name}...")
        existing_rf = self.load_latest_model('random_forest', target_name)

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

    def save_models(self, models, target_name):
        """Save trained models to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for model_name, model in models.items():
            model_path = os.path.join(self.models_dir, f"{model_name}_{target_name}_{timestamp}.pkl")
            joblib.dump(model, model_path)
            print(f"Saved {model_name} to {model_path}")

    def run(self):
        """Main training pipeline."""
        print("Loading features from DuckDB...")
        df = self.load_features()

        # 1. Regression Targets
        regression_targets = {
            'temp_max': ['temp_max'],
            'temp_min': ['temp_min'],
            'temp_mean': ['temp_mean'],      # Processed as regression
            'wind_speed_mean': ['wind_speed_mean'],
            'precipitation_mm': ['precipitation_mm'],
            'humidity_mean': ['humidity_mean'] # Processed as regression
        }

        all_results = {}

        for target_name, target_cols in regression_targets.items():
            print(f"\n{'='*60}")
            print(f"Training REGRESSORS for: {target_name}")
            print(f"{'='*60}")

            X, y = self.prepare_data(df, target_cols)
            # Check if columns exist (humidity might be missing if dbt didn't run yet)
            if y.shape[1] == 0:
                print(f"Skipping {target_name} (column not found)")
                continue

            models, results = self.train_regressors(X, y[target_cols[0]], target_name)
            all_results[target_name] = results
            self.save_models(models, target_name)

            # Print results
            print(f"\n{target_name} Results:")
            for model_name, metrics in results.items():
                print(f"  {model_name}: RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")

        # 2. Classification Target (Rain Probability)
        print(f"\n{'='*60}")
        print(f"Training CLASSIFIERS for: rain_probability")
        print(f"{'='*60}")

        # Create binary target: 1 if rain > 0.1mm, else 0
        df['is_raining'] = (df['precipitation_mm'] > 0.1).astype(int)

        X, y = self.prepare_data(df, ['is_raining'])
        models, results = self.train_classifiers(X, y['is_raining'], 'rain_probability')

        all_results['rain_probability'] = results
        self.save_models(models, 'rain_probability')

        print(f"\nrain_probability Results:")
        for model_name, metrics in results.items():
            print(f"  {model_name}: Accuracy={metrics['accuracy']:.4f}, AUC={metrics['roc_auc']:.4f}")

        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"{'='*60}")

        return all_results

if __name__ == "__main__":
    trainer = WeatherModelTrainer()
    results = trainer.run()
