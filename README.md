TEAM DETAILS:<br>
Chinar Chhabria<br>
Trisha Sheth <br>
Shreeja Brahmbhat<br>
Riya Parmar <br>
Jay Mehta<br>

# ☀️ AI-Driven Solar Inverter Failure Prediction
### HACKaMINeD 2026 — Aubergine Datasets Track

> Predict inverter shutdown and significant underperformance within a 10-day window using machine learning, surface risk scores on a live dashboard, and explain failures in natural language using Gemini AI.

---

## 📌 Problem Statement

Solar inverters can fail silently — degrading over days before a full shutdown. Plant operators have no early warning system, leading to unplanned downtime and revenue loss. This system monitors all inverters continuously, predicts which units are at risk in the next 10 days, and explains *why* in plain language so operators can act before failure occurs.

---

## 🏗️ System Architecture

```
Raw Telemetry (CSV / 5-min intervals)
            │
            ▼
┌─────────────────────────┐
│   ML Pipeline (Python)  │  ← XGBoost + Isolation Forest
│   Feature Engineering   │  ← 21 telemetry + KPI features
│   Risk Scoring          │  ← Binary + Multi-class output
└────────────┬────────────┘
             │  model.pkl / live_predictions.csv
             ▼
┌─────────────────────────┐
│   Flask REST API        │  ← /predict  /predictions  /health
│   (Python)              │  ← Input validation + error handling
└────────────┬────────────┘
             │  JSON
             ▼
┌─────────────────────────┐     ┌──────────────────────────┐
│   Dashboard (HTML/JS)   │────▶│   Gemini AI Layer        │
│   Risk cards per        │     │   Narrates failure risk  │
│   inverter + trend      │     │   Answers operator Q&A   │
│   charts                │     │   Root cause analysis    │
└─────────────────────────┘     └──────────────────────────┘
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Machine Learning | Python, XGBoost, Scikit-learn, SHAP, Isolation Forest |
| Backend API | Python, Flask, Flask-CORS |
| Frontend Dashboard | HTML, CSS, JavaScript |
| GenAI / Insight Engine | Google Gemini API, JavaScript |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Model Explainability | SHAP (SHapley Additive exPlanations) |

---

## 📁 Project Structure

```
solar-inverter-failure-prediction/
│
├── ml/
│   ├── solar_inverter_v3.ipynb     # Full ML pipeline (run this first)
│   ├── model.pkl                   # Trained XGBoost binary classifier
│   ├── model_multi.pkl             # Multi-class risk classifier
│   ├── anomaly_model.pkl           # Isolation Forest anomaly detector
│   ├── scaler.pkl                  # StandardScaler for Logistic Regression
│   ├── features.json               # Ordered feature list for API
│   ├── live_predictions.csv        # Current risk scores per inverter
│   ├── predictions.csv             # Full test set predictions
│   ├── inverter_summary.csv        # Per-inverter aggregated risk summary
│   └── shap_values.csv             # SHAP feature attributions per inverter
│
├── api/
│   ├── app.py                      # Flask REST API
│   └── requirements.txt            # Python dependencies
│
├── dashboard/
│   └── index.html                  # Frontend dashboard
│
└── README.md
```

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Node.js (optional, for local frontend serving)
- Google Gemini API key

---

### Step 1 — Run the ML Pipeline

Open `ml/solar_inverter_v3.ipynb` in Google Colab or Jupyter and run all cells top to bottom. This will:

- Load inverter telemetry from Aubergine datasets
- Engineer 21 features (9 telemetry + 12 computed KPIs)
- Train and tune XGBoost, Random Forest, and Logistic Regression
- Evaluate on a chronological hold-out test set
- Save all model artifacts and prediction CSVs to the `ml/` directory

---

### Step 2 — Install API Dependencies

```bash
cd api
pip install -r requirements.txt
```

`requirements.txt`:
```
flask
flask-cors
pandas
numpy
scikit-learn
xgboost
shap
google-generativeai
```

---

### Step 3 — Configure Gemini API Key

Set your Gemini API key as an environment variable:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Or create a `.env` file in the `api/` directory:

```
GEMINI_API_KEY=your_api_key_here
```

---

### Step 4 — Start the Flask API

```bash
cd api
python app.py
```

API will be running at `http://localhost:5000`

---

### Step 5 — Open the Dashboard

Open `dashboard/index.html` directly in your browser, or serve it locally:

```bash
cd dashboard
python -m http.server 8080
```

Then visit `http://localhost:8080`

---

## 🔌 API Endpoints

### `GET /health`
Health check — confirms API and model are loaded.

**Response:**
```json
{ "status": "ok", "model": "loaded" }
```

---

### `GET /predictions`
Returns pre-computed risk scores for all inverters based on their latest 24 hours of telemetry.

**Response:**
```json
[
  {
    "inverter_id": "P1_LT1_7",
    "risk_score": 0.934,
    "risk_level": "HIGH",
    "risk_category": "Shutdown Risk",
    "last_reading": "2024-10-13T14:00:00",
    "current_temp": 71.2,
    "power_ratio_24hr": 0.87
  }
]
```

---

### `POST /predict`
Accepts a single inverter reading and returns a real-time risk prediction.

**Request body:**
```json
{
  "temp": 68.5,
  "power": 142.0,
  "pv1_power": 138.0,
  "v_ab": 412.1,
  "v_bc": 413.4,
  "v_ca": 411.9,
  "freq": 50.01,
  "kwh_today": 620.0,
  "ambient_temp": 35.2,
  "power_ratio": 0.88,
  "temp_deviation": 9.1,
  "voltage_imbalance": 0.003,
  "temp_spike_flag": 1,
  "temp_rolling_mean": 63.4,
  "temp_rolling_std": 2.1,
  "power_rolling_mean": 148.0,
  "power_ratio_rolling_mean": 0.89,
  "voltage_imbalance_rolling": 0.002,
  "anomaly_score": 0.61,
  "hour": 13,
  "month": 7
}
```

**Response:**
```json
{
  "risk_score": 0.91,
  "risk_level": "HIGH",
  "risk_category": "Shutdown Risk",
  "fault_predicted": 1
}
```

---

## 🤖 ML Model Details

### Features (21 total)

**Telemetry (9)** — raw sensor readings broadcast every 5 minutes:
`temp`, `power`, `pv1_power`, `v_ab`, `v_bc`, `v_ca`, `freq`, `kwh_today`, `ambient_temp`

**Computed KPIs (12)** — derived metrics engineered to capture fault patterns:

| Feature | Description |
|---|---|
| `power_ratio` | Inverter output ÷ fleet median at same timestamp |
| `temp_deviation` | Inverter temp − fleet median temp |
| `voltage_imbalance` | Phase voltage spread ÷ mean voltage |
| `temp_spike_flag` | 1 if temp > 2 std dev above inverter's own history |
| `temp_rolling_mean` | 24-hour rolling average temperature |
| `temp_rolling_std` | 24-hour temperature stability |
| `power_rolling_mean` | 24-hour rolling average power output |
| `power_ratio_rolling_mean` | 24-hour rolling peer comparison ratio |
| `voltage_imbalance_rolling` | 24-hour rolling voltage imbalance |
| `anomaly_score` | Isolation Forest anomaly score (train-only fit) |
| `hour` | Hour of day (solar context) |
| `month` | Month of year (seasonality) |

### Model Performance

| Model | F1 Score | AUC | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | 0.7985 | 0.9528 | 0.69 | 0.94 | 88% |
| Random Forest | 0.8540 | 0.9802 | 0.75 | 0.98 | 92% |
| **XGBoost (selected)** | **0.9187** | **0.9958** | **0.86** | **0.99** | **96%** |

*Evaluated on chronological hold-out test set (97,443 samples, 24.9% fault rate).*

### Key Design Decisions

- **No target leakage:** Fault labels defined from raw instantaneous values only. Rolling means used as predictive features, never in label creation.
- **Chronological split:** Train (70%) / Dev (15%) / Test (15%) by timestamp. No random shuffling.
- **TimeSeriesSplit CV:** Cross-validation folds respect temporal order, preventing near-duplicate temporal samples appearing in both train and validation.
- **Primary metric — PR-AUC:** More reliable than accuracy in rare-event settings where fault events are significantly outnumbered by healthy readings.
- **Anomaly detection:** Isolation Forest fit on healthy training data only, then applied to dev/test independently to prevent leakage.
- **Multi-class output:** No Risk / Degradation Risk / Shutdown Risk for operator triage.
- **SHAP explainability:** Top 3 risk drivers computed per inverter, attached to every prediction for operator-facing decisions.
- **Final model retrained on Train + Dev** before single hold-out test evaluation.

---

## ⚠️ Known Limitations

1. The system heavily depends on the quality and completeness of data.
2. The system predicts the probability of failure, not a guaranteed outcome.
3. Sometimes the AI may generate incorrect or exaggerated explanations.
4. Security and data privacy risks.

---

## 👥 Team

| Name | Role | Component |
|---|---|---|
| **Chinar** *(Team Leader)* | ML Engineer | XGBoost pipeline, feature engineering, SHAP explainability, model evaluation |
| **Riya** | Backend + GenAI | Flask REST API, input validation, Gemini AI narration and Q&A |
| **Jay** | Backend | Flask REST API, error handling, health check endpoint |
| **Shreeja** | Frontend | Dashboard UI, inverter risk cards, trend charts |
| **Trisha** | Frontend | Dashboard UI, data visualisation, JavaScript integration |

---

## 🔮 Future Improvements

1. Plant Risk Visualisation Map
2. Color-Based Risk Indicators
3. Financial loss prediction

Built for HACKaMINeD 2026 — Aubergine Datasets Track.
# demo 
