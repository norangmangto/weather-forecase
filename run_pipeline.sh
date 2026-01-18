#!/bin/bash
set -e

echo "========================================================"
echo "🌤️  Starting Weather Prediction Pipeline"
echo "========================================================"

echo "[1/4] Ingesting Data..."
uv run src/ingest.py

echo "[2/4] transforming Data (dbt)..."
cd dbt
uv run dbt run --profiles-dir .
cd ..

echo "[3/4] Training Models (Incremental)..."
uv run src/train.py

echo "[4/4] Generating Forecast..."
uv run src/predict.py "$@"

echo "========================================================"
echo "✅ Pipeline Completed Successfully!"
echo "========================================================"
