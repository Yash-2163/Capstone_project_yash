
# 🚀 Sales Conversion Prediction Pipeline
> End‑to‑end ML pipeline for lead conversion scoring  
> Built using CRISP-ML(Q) principles: reproducible, automated, monitored.

---

## 📖 Table of Contents
1. [Business & Data Understanding](#1-business--data-understanding)  
2. [Data Preparation](#2-data-preparation)  
3. [Model Engineering](#3-model-engineering)  
4. [Model Evaluation](#4-model-evaluation)  
5. [Deployment & Inference](#5-deployment--inference)  
6. [Monitoring & Maintenance](#6-monitoring--maintenance)  
7. [Architecture Diagram](#7-architecture-diagram)  
8. [Setup & Deployment Instructions](#8-setup--deployment-instructions)  
9. [Troubleshooting & Lessons Learned](#9-troubleshooting--lessons-learned)  
10. [Future Improvements](#10-future-improvements)  
11. [References](#11-references)  

---

## 1. Business & Data Understanding

**Objective:**  
Predict which leads are most likely to convert so sales teams can focus on “hot leads” and improve conversion rates from ~30% to 80%.

**Data Sources:**  
- CSV uploads ingested via S3  
- Processed via AWS Glue  
- Stored in Redshift  

**Data Sample:**  
| Column            | Type    | % Missing | Notes                      |
|-------------------|---------|-----------|----------------------------|
| Prospect ID       | string  | 0%        | Unique lead identifier     |
| Converted         | int     | 0%        | Target variable (0/1)      |
| TotalVisits       | float   | ~2%       | Skewed, log transformed    |
| Lead Source       | string  | ~5%       | Simplified via frequency   |
| Tags, Lead Profile| string  | > 50%     | Dropped                    |

📸 _Insert: missingness heatmap_

---

## 2. Data Preparation

### 2.1 Missing Value Handling  
- Numeric: median imputation  
- Categorical: mode fill or “Unknown”  
- High-missing: dropped (Tags, Lead Profile, etc.)

### 2.2 Skew & Outlier Treatment  
- Applied `log1p` on skewed features  
- Removed outliers using IQR on `TotalVisits`, `Page Views`

📸 _Insert: before/after histogram_

### 2.3 Encoding & Feature Selection  
- Yes/No mapped to 1/0  
- Rare categories grouped as “Other”  
- OneHotEncoding for nominal categories  

### 2.4 Pipeline  
All transformations wrapped in a single `sklearn.Pipeline`.

📸 _Insert: preprocessing pipeline diagram_

---

## 3. Model Engineering

### 3.1 Split & Strategy  
- 80/20 stratified train-test split  
- Trained 3 models:  
  - Logistic Regression (baseline)  
  - Random Forest ✅ Best  
  - XGBoost  

### 3.2 Training Environment  
- Training run in SageMaker Studio  
- Tracked in MLflow hosted on SageMaker  

📸 _Insert: MLflow UI screenshot_

---

## 4. Model Evaluation

| Model              | Accuracy | F1 Score | ROC AUC |
|-------------------|----------|----------|---------|
| LogisticRegression| 0.72     | 0.66     | 0.79    |
| RandomForest       | 0.83     | 0.80     | 0.89    |
| XGBoost            | 0.83     | 0.75     | 0.87    |

📸 _Insert: confusion matrix, ROC curves_

---

## 5. Deployment & Inference

### 5.1 Flask API + Ngrok  
- `/predict` route accepts uploaded CSV  
- Returns JSON predictions  
- Ngrok used to expose API for testing/demo

### 5.2 Web Interface  
- Streamlit-style simple web UI planned

📸 _Insert: screenshot of prediction result_

---

## 6. Monitoring & Maintenance

### 6.1 Drift Detection  
- Uses `ks_2samp` from SciPy  
- Triggered via Airflow DAG  
- Results saved as JSON + HTML via Evidently

### 6.2 Airflow Orchestration  
- MWAA triggers:  
  - Glue Crawler  
  - Glue ETL Job  
  - Preprocessing + training  
  - Drift detection  
  - Conditional retraining

📸 _Insert: Airflow DAG screenshot_

### 6.3 MLflow Registry  
- Auto-registration of best model  
- Stored in S3 and SageMaker registry  
- Version tracked and transitioned to staging

---

## 7. Architecture Diagram

📸 _Insert: AWS ML architecture diagram_

- S3 → Glue Crawler → Glue Job → Redshift  
- Redshift → SageMaker → MLflow  
- MLflow → S3 + Flask API  
- Flask + Ngrok → Airflow  
- Airflow → Drift detection + retraining

---

## 8. Setup & Deployment Instructions

### Prerequisites
- AWS CLI configured  
- Python 3.9+  
- Docker + Docker Compose  

### AWS Setup
- Create IAM roles for Glue, SageMaker, Redshift  
- Set up:  
  - S3 buckets (raw + processed)  
  - Redshift cluster  
  - Glue Crawler: `crawlerdb`  
  - Glue Job: `ETLJOB-Creation`  

### Local Setup

```bash
# Flask
cd backend/
python predict.py

# Ngrok
ngrok authtoken YOUR_TOKEN
ngrok http 5000
```

### MWAA Setup
- Upload DAGs to `s3://<your-bucket>/dags/`
- Add requirements.txt with:
  ```
  boto3
  apache-airflow-providers-amazon
  ```
- DAG ID: `lead_scoring_pipeline_nocopy`

---

## 9. Troubleshooting & Lessons Learned

| Issue                                 | Resolution                                |
|--------------------------------------|-------------------------------------------|
| `KeyError: Converted`                | Standardized column casing                |
| Flask pickle load error              | Moved custom classes to top-level scope   |
| MLflow URL unreachable from MWAA     | Used try/except fallback logging          |
| Redshift permission error            | Added IAM access + schema permission      |

---

## 10. Future Improvements

- Deploy model via **SageMaker Endpoint**  
- CI/CD for retraining + scoring  
- Slack alerts for data drift  
- Streamlit UI with authentication  
- Live CRM integration

---

## 11. References

- [CRISP-ML(Q) Framework](https://github.com/alan-turing-institute/crisp-ml)  
- [AWS Glue](https://aws.amazon.com/glue/)  
- [MLflow](https://mlflow.org)  
- [Evidently AI](https://evidentlyai.com)  
- [Airflow on AWS (MWAA)](https://docs.aws.amazon.com/mwaa/latest/userguide/what-is-mwaa.html)

---

Generated: 2025-07-21  
Author: Your Name  
