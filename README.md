# 🚗⚡ EV Charging Demand Forecasting System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-orange.svg)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)]([https://streamlit.io](https://ev-charging-demand-forecasting-and-simulator-t9uwyjxiu5hzkjqln.streamlit.app/))
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black.svg)](https://black.readthedocs.io)

> **Advanced Machine Learning System for Electric Vehicle Charging Infrastructure Optimization**

A comprehensive, production-ready machine learning pipeline that forecasts hourly EV charging demand at candidate sites to optimize charger placement and operations planning. Built with modern MLOps practices and designed for scalability. Try the project at : https://ev-charging-demand-forecasting-and-simulator-t9uwyjxiu5hzkjqln.streamlit.app

## 🎯 Project Overview

### Business Problem
With the rapid adoption of electric vehicles, strategic placement of charging infrastructure is critical for:
- **Grid Load Management**: Preventing overload during peak hours
- **Revenue Optimization**: Maximizing utilization at high-demand locations  
- **User Experience**: Reducing wait times and improving accessibility
- **Investment Planning**: Data-driven decisions for infrastructure expansion

### Technical Solution
This system implements a sophisticated time-series forecasting pipeline using:
- **Feature Engineering**: Temporal patterns, lag variables, rolling statistics
- **Machine Learning**: XGBoost with hyperparameter optimization
- **Model Validation**: Time-based cross-validation and performance monitoring
- **API Deployment**: Production-ready FastAPI service with comprehensive documentation
  - **Visualization**: Enhanced Interactive Streamlit dashboard with:
    - Real-time network monitoring with live metrics
    - Dynamic demand pattern analysis with time series forecasting
    - State-wise infrastructure and utilization insights
    - Advanced geospatial visualizations and heatmaps
    - Customizable date ranges and interactive filters
    - Responsive design with modern UI components

### Dashboard Features
- **Network Overview**:
  - Live metrics with trend indicators
  - Active charging sites tracking
  - Network utilization analytics
  - Peak hour demand analysis
  - Day-over-day performance comparison
  
- **Demand Analytics**:
  - Interactive time series visualizations
  - Hourly demand heatmaps
  - Weekend vs Weekday patterns
  - Peak hour identification
  - State-wise demand distribution

- **Performance Monitoring**:
  - Real-time data updates
  - Custom CSS styling with animations
  - Mobile-responsive design
  - Efficient data caching
  - Interactive data exploration

## 🏗️ System Architecture```mermaid
graph TD
    A[Raw Charging Data] --> B[Data Pipeline]
    B --> C[Feature Engineering]
    C --> D[Model Training]
    D --> E[Model Validation]
    E --> F[Model Registry]
    F --> G[FastAPI Service]
    F --> H[Streamlit Dashboard]
    G --> I[Predictions API]
    H --> J[Interactive Visualizations]
    
    K[Monitoring] --> G
    K --> H
    L[CI/CD Pipeline] --> G
    L --> H
```

## 📁 Project Structure

```
ev-charging-demand-forecast/
├── 📊 data/
│   ├── raw/                    # Raw charging session data
│   ├── processed/              # Cleaned and aggregated data
│   ├── synthetic_generator.py  # Synthetic data generation
│   └── data_validation.py      # Data quality checks
├── 📓 notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_baseline_xgboost.ipynb
│   ├── 03_advanced_models.ipynb
│   └── 04_model_evaluation.ipynb
├── 📦 ev_forecast/
│   ├── api/
│   │   ├── app.py              # FastAPI application
│   │   ├── models.py           # Pydantic models
│   │   └── dependencies.py     # API dependencies
│   ├── models/
│   │   ├── base.py             # Base model interface
│   │   ├── xgboost_model.py    # XGBoost implementation
│   │   ├── lstm_model.py       # LSTM implementation
│   │   └── ensemble.py         # Model ensemble
│   ├── features/
│   │   ├── temporal.py         # Time-based features
│   │   ├── spatial.py          # Location-based features
│   │   └── engineering.py      # Feature engineering pipeline
│   ├── utils/
│   │   ├── config.py           # Configuration management
│   │   ├── logging.py          # Logging setup
│   │   └── monitoring.py       # Model monitoring
│   ├── data_pipeline.py        # Data processing pipeline
│   └── training.py             # Model training orchestration
├── 📱 dashboard/
│   ├── app.py                  # Streamlit application
│   ├── components/             # Reusable UI components
│   └── pages/                  # Multi-page dashboard
├── 🧪 tests/
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── conftest.py             # Test configuration
├── 🚀 deployment/
│   ├── docker-compose.yml      # Multi-container deployment
│   ├── k8s/                    # Kubernetes manifests
│   └── terraform/              # Infrastructure as code
├── 📋 docs/
│   ├── api_documentation.md    # API reference
│   ├── model_documentation.md  # Model details
│   └── deployment_guide.md     # Deployment instructions
├── .github/
│   └── workflows/              # CI/CD pipelines
├── requirements/
│   ├── base.txt                # Base dependencies
│   ├── dev.txt                 # Development dependencies
│   └── prod.txt                # Production dependencies
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Local development setup
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12
- Docker (optional)
- Git

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/adityagit94/EV-Charging-Demand-Forecasting-and-Simulator.git
cd EV-Charging-Demand-Forecasting-and-Simulator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements/dev.txt
```

### 2. Data Generation & Training
```bash
# Generate synthetic dataset
python -m ev_forecast.data.synthetic_generator --sites 10 --days 90

# Run exploratory analysis
jupyter notebook notebooks/01_exploratory_analysis.ipynb

# Train baseline model
python -m ev_forecast.training --model xgboost --config configs/baseline.yaml

# Evaluate model performance
python -m ev_forecast.evaluation --model-path models/xgboost_v1.joblib
```

### 3. API Service
```bash
# Start FastAPI server
uvicorn ev_forecast.api.app:app --reload --port 8000

# View API documentation
open http://localhost:8000/docs
```

### 4. Enhanced Dashboard
```bash
# Launch Streamlit dashboard
streamlit run dashboard/app.py

# Access dashboard
open http://localhost:8501
```

The dashboard provides a comprehensive overview of the EV charging network with:
- Real-time monitoring of active charging sites
- Dynamic demand pattern analysis with interactive visualizations
- Advanced metrics with trend indicators
- Customizable date ranges and filters
- Responsive design with modern UI components
- State-wise analytics with detailed insights
- Peak hour identification and analysis
- Weekend vs Weekday pattern comparison

### 5. Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Services available at:
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

## 🔬 Model Performance

### Baseline XGBoost Results
| Metric | Value | Benchmark |
|--------|-------|-----------|
| MAE | 2.34 sessions/hour | ±0.15 |
| RMSE | 3.78 sessions/hour | ±0.22 |
| MAPE | 12.5% | <15% |
| R² Score | 0.847 | >0.80 |

### Advanced Models
- **LSTM**: Captures long-term temporal dependencies
- **Ensemble**: Combines multiple models for improved accuracy
- **AutoML**: Automated hyperparameter optimization

## 📊 Key Features

### Data Engineering
- **Temporal Features**: Hour, day, week, month patterns
- **Lag Variables**: 1h, 24h, 168h (weekly) lags
- **Rolling Statistics**: Moving averages and standard deviations
- **Spatial Features**: Location-based demand patterns
- **Weather Integration**: Temperature, precipitation effects

### Model Capabilities
- **Multi-horizon Forecasting**: 1-hour to 7-day predictions
- **Uncertainty Quantification**: Prediction intervals
- **Feature Importance**: SHAP explainability
- **Online Learning**: Incremental model updates
- **A/B Testing**: Model comparison framework

### Production Features
- **Model Versioning**: MLflow integration
- **Data Validation**: Great Expectations
- **Monitoring**: Prometheus metrics
- **Alerting**: Slack/email notifications
- **Caching**: Redis for fast predictions
- **Rate Limiting**: API protection

### Dashboard Features (New)
- **Real-time Monitoring**:
  - Live metrics with trend indicators
  - Active site tracking
  - Network utilization analytics
  - Peak demand monitoring
  
- **Interactive Visualizations**:
  - Time series plots with forecasting
  - Dynamic heatmaps
  - State-wise comparisons
  - Custom date range selection
  
- **Enhanced UI/UX**:
  - Modern card-based design
  - Hover animations and transitions
  - Mobile-responsive layout
  - Color-coded indicators
  - Intuitive navigation
  
- **Performance Optimization**:
  - Efficient data caching
  - Optimized query patterns
  - Responsive data loading
  - Smart data aggregation

## 🧪 Testing & Quality Assurance

```bash
# Run all tests
pytest tests/ -v --cov=ev_forecast --cov-report=html

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/performance/   # Performance tests

# Code quality checks
black ev_forecast/ tests/           # Code formatting
flake8 ev_forecast/ tests/          # Linting
mypy ev_forecast/                   # Type checking
```

## 📈 Monitoring & Observability

### Model Monitoring
- **Data Drift Detection**: Statistical tests for input distribution changes
- **Performance Degradation**: Automated alerts for accuracy drops
- **Feature Importance Tracking**: Monitor feature contributions over time

### System Monitoring
- **API Metrics**: Response time, error rates, throughput
- **Resource Usage**: CPU, memory, disk utilization
- **Business Metrics**: Prediction accuracy, user engagement

## 🔧 Configuration

The system uses hierarchical configuration management:

```yaml
# configs/production.yaml
model:
  name: "xgboost"
  hyperparameters:
    max_depth: 8
    learning_rate: 0.1
    n_estimators: 500
  
data:
  features:
    temporal: true
    spatial: true
    weather: false
  
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
```

## 🚀 Deployment Options

### Local Development
- Direct Python execution
- Jupyter notebooks for experimentation
- Hot reload for rapid development

### Production Deployment
- **Docker**: Containerized services
- **Kubernetes**: Scalable orchestration
- **AWS/GCP/Azure**: Cloud deployment
- **CI/CD**: Automated testing and deployment

## 📚 Documentation

- [API Documentation](docs/api_documentation.md) - Complete API reference
- [Model Documentation](docs/model_documentation.md) - Technical model details
- [Deployment Guide](docs/deployment_guide.md) - Production deployment
- [Contributing Guide](CONTRIBUTING.md) - Development guidelines

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **XGBoost Team** for the excellent gradient boosting framework
- **FastAPI** for the modern Python web framework
- **Streamlit** for the intuitive dashboard framework
- **Open Source Community** for the foundational tools

## 📞 Contact

- **Author**: Aditya Prakash
- **Email**: aditya_2312res46@iitp.ac.in
- **GitHub**: [@adityagit94](https://github.com/adityagit94)

---

⭐ **Star this repository if it helped you!**

*Built with ❤️ for sustainable transportation and smart grid infrastructure*
