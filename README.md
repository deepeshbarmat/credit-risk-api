# Credit Risk Prediction API (MLOps Project)

A machine learning service that trains, tracks, packages, and serves a credit-risk prediction model using modern MLOps practices.

This project demonstrates **end-to-end ML system design**, not just model training.

---

## Project Highlights

* Reproducible ML training pipeline
* MLflow experiment tracking and model registry
* FastAPI inference service
* Dockerized deployment
* GitHub Actions CI pipeline
* Automated API testing with pytest
* Modular ML project structure

This repository simulates how ML systems are built in real production environments.

---

## System Architecture

```
Dataset → Training Pipeline → MLflow Tracking
                                ↓
                         Registered Model
                                ↓
                           Prediction Module
                                ↓
                            FastAPI Service
                                ↓
                               Docker
                                ↓
                         GitHub Actions CI
```

---

## Dataset

German Credit Risk dataset

Kaggle:
https://www.kaggle.com/datasets/benjaminmcgregor/german-credit-data-set-with-credit-risk

Local Dataset location:
```
data/german_credit_data.csv
```

---

## Repository Structure

```
credit-risk-api/
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── utils.py
│   ├── features/
│   │   └── build_features.py
│   ├── api/
│   │   └── app.py
│   └── models/
│       └── model.py
│
├── notebooks/
│   ├── eda.ipynb
│
├── tests/
│   ├── test_api.py
│   └── test_train.py
│
├── models/
├── data/
│   └── german_credit_data.csv
│
├── Dockerfile
├── requirements.txt
├── requirements-api.txt
├── requirements-ci.txt
├── .flake8
│
└── .github/workflows/ci.yml
```

---

## Training Pipeline

The training script:

```
src/train.py
```

Capabilities:

* Data loading and preprocessing
* Feature pipeline creation
* Runtime model selection
* MLflow logging
* Model registration
* Local pipeline persistence

Supported models:

* Logistic Regression
* XGBoost

Run training:

```
python src/train.py --model xgb
```

Metrics logged to MLflow:

* Accuracy
* ROC AUC
* F1 score

---

## Prediction Module

```
src/predict.py
```

Responsibilities:

* Load trained pipeline
* Accept structured input
* Return prediction and probability

Example output:

```
{
  "prediction": "Low Risk",
  "probability": 0.18
}
```

---

## FastAPI Service

```
src/api/app.py
```

Endpoints:

### Health Check

```
GET /health
```

### Predict

```
POST /predict
```

Run locally:

```
uvicorn src.api.app:app --reload
```

Swagger UI:

```
http://localhost:8000/docs
```

---

## Docker

Build container:

```
docker build -t credit-risk-api .
```

Run container:

```
docker run -p 8000:8000 credit-risk-api
```

---

## CI Pipeline

GitHub Actions workflow:

```
.github/workflows/ci.yml
```

Pipeline performs:

* Dependency installation
* Model training test
* Code linting
* API tests with pytest

Run tests locally:

```
pytest tests/
```

---

## Technology Stack

Machine Learning:

* scikit-learn
* XGBoost
* pandas
* LogisticRegression

MLOps:

* MLflow
* Docker
* GitHub Actions

API:

* FastAPI
* Pydantic
* Uvicorn

Testing:

* pytest

---

## Why This Project Matters

This project demonstrates the ability to:

* Build ML pipelines
* Track experiments
* Register models
* Serve ML models via API
* Containerize ML systems
* Implement CI pipelines
* Structure production-ready ML code

These skills are directly relevant to:

* ML Engineer roles
* MLOps Engineer roles
* AI Platform Engineer roles
* ML DevOps roles

---

## How to Run the Project

Train model:

```
python src/train.py --model xgb
```

Run API:

```
uvicorn src.api.app:app --reload
```

Run Docker:

```
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

Run tests:

```
pytest tests/
```

---

## Future Improvements

Possible extensions:

* Model version promotion workflow
* Monitoring and logging
* Kubernetes deployment
* Feature store integration
* Data validation pipeline
* RAG-based decision explanation service
* CI/CD deployment pipeline

---

## License

This is a learning project.

---
