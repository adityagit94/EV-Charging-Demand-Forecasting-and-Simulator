#!/bin/bash
echo "1) Generate synthetic data"
python data/synthetic_generator.py
echo "2) Train baseline model"
python src/models/train_xgboost.py
echo "3) Start API (background)"
uvicorn src.api.app:app --reload --port 8000 &>/dev/null &
echo "API running at http://127.0.0.1:8000"
echo "4) To run dashboard: streamlit run dashboard/app.py"
