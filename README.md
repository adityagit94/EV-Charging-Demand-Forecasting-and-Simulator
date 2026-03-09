# 🚗⚡ EV Charging Demand Forecasting System

<p align="center">
<b>Machine Learning System for Forecasting EV Charging Demand and Infrastructure Planning</b>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![PyTest](https://img.shields.io/badge/Tests-PyTest-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Code Style](https://img.shields.io/badge/Code%20Style-Black-black)

</p>

<p align="center">

🔗 <b>Live Demo</b>  
https://ev-charging-demand-forecasting-and-simulator-t9uwyjxiu5hzkjqln.streamlit.app

</p>

---

# 📑 Table of Contents

- Project Overview
- System Architecture
- Machine Learning Pipeline
- Dashboard Preview
- Project Structure
- Quick Start
- API Example
- Technology Stack
- Testing
- Future Improvements

---

# 🌍 Project Overview

Electric vehicle adoption is accelerating globally, creating significant demand for reliable charging infrastructure. Planning charging networks requires **predicting where and when charging demand will occur.**

This project implements a **full machine learning pipeline** that forecasts hourly EV charging demand and provides an **interactive dashboard for analyzing network utilization and infrastructure planning.**

The system demonstrates a realistic ML workflow including:

• Synthetic dataset generation  
• Data preprocessing and validation  
• Temporal feature engineering  
• Machine learning forecasting  
• API-based model serving  
• Interactive analytics dashboard

---

# ✨ Key Highlights

✔ End-to-end ML pipeline

✔ Feature engineered time-series forecasting

✔ FastAPI prediction service

✔ Interactive Streamlit analytics dashboard

✔ Dockerized deployment

✔ Clean modular Python architecture

✔ Reproducible ML workflow

---

# 🏗 System Architecture

The system is designed using a modular ML pipeline architecture.

```mermaid
flowchart LR

classDef data fill:#E3F2FD,stroke:#1E88E5,color:#000
classDef feature fill:#E8F5E9,stroke:#43A047,color:#000
classDef model fill:#FFF3E0,stroke:#FB8C00,color:#000
classDef serve fill:#F3E5F5,stroke:#8E24AA,color:#000
classDef ui fill:#FCE4EC,stroke:#D81B60,color:#000

subgraph DATA_LAYER
A[Raw Charging Data]
B[Data Validation]
C[Hourly Aggregation]
end

subgraph FEATURE_ENGINEERING
D[Feature Engineering]
E[Lag Features]
F[Rolling Statistics]
G[Temporal Encoding]
end

subgraph MODEL_TRAINING
H[XGBoost Model Training]
I[Model Evaluation]
J[Saved Model Artifact]
end

subgraph MODEL_SERVING
K[FastAPI Service]
L[Prediction Endpoint]
end

subgraph ANALYTICS
M[Streamlit Dashboard]
N[Interactive Charts]
O[Demand Analysis]
end

A:::data --> B:::data --> C:::data
C --> D:::feature

D --> E:::feature
D --> F:::feature
D --> G:::feature

E --> H:::model
F --> H:::model
G --> H:::model

H --> I:::model --> J:::model

J --> K:::serve
K --> L:::serve

J --> M:::ui
M --> N:::ui
M --> O:::ui

```

The diagram visually represents the **data → features → model → serving → analytics workflow** used in the project.

---

# 🧠 Machine Learning Pipeline

The forecasting workflow follows a structured ML lifecycle:

### 1️⃣ Data Generation
Synthetic EV charging sessions simulate realistic demand patterns across charging stations.

### 2️⃣ Data Pipeline
The preprocessing pipeline performs:

• schema validation  
• duplicate removal  
• invalid value handling  
• hourly demand aggregation

### 3️⃣ Feature Engineering

Key engineered features:

• hour / weekday / month indicators  
• cyclical time encoding (sin / cos)  
• lag demand variables  
• rolling demand averages  
• peak-hour indicators

### 4️⃣ Model Training

A gradient boosting model (**XGBoost**) is trained to forecast hourly charging demand.

### 5️⃣ Model Serving

The trained model is deployed using **FastAPI** for real-time predictions.

### 6️⃣ Visualization

The Streamlit dashboard allows **interactive analysis of charging demand patterns.**

---

# 📊 Dashboard Preview

The dashboard enables exploration of EV charging demand patterns.

### Network Overview

• Total sessions across the network

• Utilization metrics

• Active charging sites

• Peak demand hours


### Demand Analysis

• Hourly demand heatmaps

• Weekday vs weekend comparison

• State level demand aggregation

• Interactive time series charts


### Interactive Features

• Date range filters

• Dynamic visualizations

• Responsive UI

• Cached data loading

---

# 📁 Project Structure

```
EV-Charging-Demand-Forecasting/

├── data/
│   └── synthetic_generator.py
│
├── ev_forecast/
│   ├── data_pipeline.py
│   ├── features.py
│   ├── training.py
│   └── utils/
│
├── api/
│   └── app.py
│
├── dashboard/
│   ├── app.py
│   └── components/
│
├── notebooks/
│   └── baseline_xgboost.ipynb
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# 🚀 Quick Start

## Clone Repository

```bash
git clone https://github.com/adityagit94/EV-Charging-Demand-Forecasting-and-Simulator.git
cd EV-Charging-Demand-Forecasting-and-Simulator
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Generate Dataset

```bash
python data/synthetic_generator.py
```

## Train Model

```bash
python -m ev_forecast.training
```

## Start API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

API docs:

```
http://localhost:8000/docs
```

## Run Dashboard

```bash
streamlit run dashboard/app.py
```

---

# 🔌 Example Prediction Request

```json
POST /predict
{
  "site_id": "site_01",
  "timestamp": "2026-05-01T10:00:00"
}
```

Response

```json
{
  "predicted_sessions": 8.3
}
```

---

# 🛠 Technology Stack

| Layer | Technology |
|------|------|
| Programming | Python |
| Data Processing | Pandas / NumPy |
| Machine Learning | XGBoost |
| API | FastAPI |
| Visualization | Streamlit + Plotly |
| Containerization | Docker |
| Testing | PyTest |

---

# 🧪 Testing

Run the full test suite:

```bash
pytest
```

---

# 📈 Future Improvements

Potential extensions include:

• Weather-aware demand forecasting  
• Charger placement optimization simulator  
• Geospatial demand modeling  
• Deep learning time-series forecasting  

---

# ⭐ Support

If you found this project useful, consider **starring the repository**.

---

# 📄 License

MIT License

