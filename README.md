<div align="center">

# 📡 Telco Customer Churn Predictor

**An end-to-end ML system that identifies at-risk telecom customers — from raw data to a live REST API and interactive dashboard.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-orange)](https://xgboost.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?logo=mlflow)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://docker.com)

<!-- Add a demo GIF here once available -->
<!-- ![Demo](assets/demo.gif) -->

</div>

---

## 🧭 What This Project Does

Telecom companies lose significant revenue when customers churn. This project builds a **production-style ML pipeline** that:

1. Ingests raw customer data (demographics, usage, billing)
2. Engineers features and trains an XGBoost classifier
3. Tracks every experiment with MLflow
4. Serves real-time predictions via a **Dockerized FastAPI backend**
5. Exposes an interactive Streamlit dashboard for business users

> **Design choice:** The model is tuned for high recall (~0.83) over precision — it's more costly to miss a churner than to flag a false positive.

---

## 📊 Model Performance

| Metric    | Score |
|-----------|-------|
| Precision | ~0.48 |
| Recall    | ~0.83 |
| F1 Score  | ~0.61 |
| ROC-AUC   | ~0.83 |

The high recall / moderate precision trade-off is intentional — in churn use cases, a missed churner (false negative) costs more than a wrongly flagged loyal customer (false positive). Retention outreach is cheap; losing a customer is not.

---

## 🏗️ Architecture

```
Raw CSV → Preprocessing → Feature Engineering → XGBoost Training
              ↓                                        ↓
       Feature columns                          MLflow Tracking
       saved for reuse                               ↓
              └─────────────── FastAPI ──────────────┘
                               (Docker)
                                   ↓
                            Streamlit UI
```

**Train-serve consistency** is enforced by saving the exact feature column list at training time and reloading it at inference — preventing silent schema drift bugs.

---

## 📂 Project Structure

```
ml/
├── data/
│   └── raw/                  # Source dataset (Telco-Customer-Churn.csv)
├── scripts/
│   └── run_pipeline.py       # End-to-end training entrypoint
├── src/
│   ├── data/                 # Data loading & preprocessing
│   ├── features/             # Feature engineering (binary + one-hot encoding)
│   ├── models/               # XGBoost training logic
│   ├── serving/              # Inference pipeline
│   └── app/
│       ├── app.py            # FastAPI backend
│       └── main.py           # Streamlit frontend
├── Dockerfile                # Docker config for FastAPI backend
├── mlruns/                   # MLflow experiment artifacts
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### Option A — Local (venv)

```bash
# 1. Clone the repo
git clone https://github.com/your-username/telco-churn-predictor.git
cd telco-churn-predictor

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Option B — Docker (recommended for the API)

```bash
# Build the image
docker build -t telco-churn-app .

# Run the container
docker run -p 8000:8000 telco-churn-app
```

Swagger docs will be available at `http://127.0.0.1:8000/docs`

---

## ▶️ Usage

### 1. Train the model
```bash
python scripts/run_pipeline.py \
  --input data/raw/Telco-Customer-Churn.csv \
  --target Churn
```
MLflow will log metrics, parameters, and the trained model artifact automatically.

### 2. Start the API

**With Docker (recommended):**
```bash
docker build -t telco-churn-app .
docker run -p 8000:8000 telco-churn-app
```

**Without Docker:**
```bash
uvicorn src.app.app:app --reload
```

→ Swagger UI: `http://127.0.0.1:8000/docs`

### 3. Launch the dashboard
```bash
streamlit run src/app/main.py
```

→ Opens at `http://localhost:8501`

---

## 🐳 Docker

The FastAPI backend is fully containerized. Streamlit containerization is in progress (see Roadmap).

```bash
# Build
docker build -t telco-churn-app .

# Run
docker run -p 8000:8000 telco-churn-app

# Run in background
docker run -d -p 8000:8000 --name churn-api telco-churn-app

# Stop
docker stop churn-api
```

---

## 🧩 Key Engineering Decisions

| Decision | Why |
|----------|-----|
| **Saved feature columns** | Prevents train-serve skew — inference uses the exact schema seen at training |
| **MLflow tracking** | Reproducible experiments; easy to compare runs and load artifacts |
| **FastAPI + Streamlit split** | Clean separation of backend logic from UI; API can be consumed independently |
| **Recall-optimized threshold** | Aligns with business cost asymmetry in churn scenarios |
| **Dockerized API** | Consistent runtime environment; eliminates "works on my machine" issues |

---

## 🚧 Roadmap

- [x] Dockerize FastAPI backend
- [ ] Dockerize Streamlit frontend
- [ ] Prediction probability visualization in the dashboard
- [ ] Add SHAP-based feature importance explanations
- [ ] Deploy on Render / AWS (EC2 or Lambda)
- [ ] Improve precision via threshold tuning or cost-sensitive learning
- [ ] Add authentication to the API

---

## 👨‍💻 Author

**Sonu Kumar** — AI/ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github)](https://github.com/your-username)

---

<div align="center">
  If this project helped you, consider giving it a ⭐ — it helps others find it!
</div>