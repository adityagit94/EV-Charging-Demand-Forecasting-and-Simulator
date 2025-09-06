# EV Charging Demand Forecasting — Project Summary

**Objective:** Forecast hourly EV charging demand at candidate sites to support data-driven charger placement.

**Approach:** Synthetic dataset → feature engineering (lags, rolling statistics, temporal encodings) → XGBoost baseline → SHAP explainability → API + dashboard demo.

**Deliverables in this repo:**
- Synthetic data generator (data/synthetic_generator.py)
- Baseline notebook (notebooks/02_baseline_xgboost.ipynb)
- Feature modules (src/features.py, src/data_pipeline.py)
- XGBoost training script (src/models/train_xgboost.py)
- FastAPI demo (src/api/app.py)
- Streamlit dashboard (dashboard/app.py)
- Dockerfile and requirements.txt

**How to run (fast):**
1. `pip install -r requirements.txt`
2. `python data/synthetic_generator.py`
3. `jupyter notebook notebooks/02_baseline_xgboost.ipynb`
4. `python src/models/train_xgboost.py` (to train and save model)
5. `uvicorn src.api.app:app --reload --port 8000`
6. `streamlit run dashboard/app.py`

**Contact / Next steps:**
Adapt the data loader to accept real charging logs. Add spatial features (POIs) and try LSTM/TFT models for improved short-term forecasting.
