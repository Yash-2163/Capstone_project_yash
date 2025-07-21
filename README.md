
# 🚀 Lead Conversion Prediction Pipeline
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
.......

---

## 2. Data Preparation

### 2.1 Missing Value Handling  
- Numeric: median imputation  
- Categorical: mode fill or “Unknown”  
- High-missing: dropped (Tags, Lead Profile, etc.)

### 2.2 Skew & Outlier Treatment  
- Applied `log1p` on skewed features  
- Removed outliers using IQR on `TotalVisits`, `Page Views`

### 2.3 Encoding & Feature Selection  
- Yes/No mapped to 1/0  
- Rare categories grouped as “Other”  
- OneHotEncoding for nominal categories  

### 2.4 Pipeline  
All transformations wrapped in a single `sklearn.Pipeline`.

<img width="921" height="614" alt="image" src="https://github.com/user-attachments/assets/b9a57274-d243-429d-8342-8a8aca7918fd" />


---

## 3. Model Engineering

### 3.1 Split & Strategy  
- 80/20 stratified train-test split  
- Trained multiple models and choosen the best one model:  
  - Decision Tree (baseline)
  - LightGBM (Best Model)
### 3.2 Training Environment  
- Training run in SageMaker Studio
- <img width="1919" height="1078" alt="Screenshot 2025-07-21 042606" src="https://github.com/user-attachments/assets/1fbd75f4-bca5-413c-84e1-4291366be759" />

- Tracked in MLflow hosted on SageMaker  
<img width="1919" height="1079" alt="Screenshot 2025-07-20 074501" src="https://github.com/user-attachments/assets/6b0978c7-7817-4614-bda9-ac91b7f5f463" />


---

## 4. Model Evaluation

Model Comparison:

 | Model               | Accuracy | Precision | Recall   | F1 Score | ROC AUC  |
|---------------------|---------:|----------:|----------|---------:|---------:|
| LightGBM            | 0.882756 |  0.861030 |  0.829588 | 0.845017 | 0.947484 |
| Gradient Boosting   | 0.872655 |  0.850834 |  0.811798 | 0.830858 | 0.941121 |
| Random Forest       | 0.873016 |  0.848249 |  0.816479 | 0.832061 | 0.939088 |
| Extra Trees         | 0.864358 |  0.837891 |  0.803371 | 0.820268 | 0.923967 |
| Logistic Regression | 0.862554 |  0.840436 |  0.794007 | 0.816562 | 0.921199 |
| Decision Tree       | 0.830447 |  0.779963 |  0.779963 | 0.779963 | 0.825937 |

---

## 5. Deployment & Inference

### 5.1 Flask API + Ngrok  
- `/predict` route accepts uploaded CSV  
- Returns JSON predictions  
- Ngrok used to expose API for testing/demo

### 5.2 Web Interface  
- Streamlit-style simple web UI planned
<img width="1919" height="1079" alt="Screenshot 2025-07-20 170602" src="https://github.com/user-attachments/assets/63566dd7-d04b-4859-8aa2-108fdebff736" />

<img width="1917" height="1079" alt="Screenshot 2025-07-21 104238" src="https://github.com/user-attachments/assets/6e4f8104-0b8c-4574-bd85-d0676957ebb9" />


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

<img width="1918" height="1079" alt="Screenshot 2025-07-20 134552" src="https://github.com/user-attachments/assets/053d83f9-4e41-47af-a3a4-65c77cbb49a0" />


### 6.3 MLflow Registry  
- Auto-registration of best model  
- Stored in S3 and SageMaker registry  
- Version tracked and transitioned to staging
- <img width="1909" height="1079" alt="Screenshot 2025-07-20 042645" src="https://github.com/user-attachments/assets/9971cacc-4030-4214-bb33-e8cc4a45e664" />


---

## 7. Architecture Diagram

<img width="2000" height="465" alt="image" src="https://github.com/user-attachments/assets/c56d9abb-0608-4239-aeff-075221245848" />


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
Author: Yash Rajput
