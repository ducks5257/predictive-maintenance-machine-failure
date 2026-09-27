# Predictive Maintenance & Intelligent Machine Failure Prediction System

An end-to-end machine learning system for monitoring industrial machines, predicting near-term failures, detecting abnormal sensor behaviour, estimating remaining useful life, and explaining failure predictions using SHAP.

## Live Demo

Frontend:  
https://predictive-machine-maintainance.netlify.app/

Backend API:  
https://predictive-maintenance-machine-failure.onrender.com

Swagger API Docs:  
https://predictive-maintenance-machine-failure.onrender.com/docs

---

## Project Overview

Industrial machines continuously generate sensor readings such as voltage, rotation, pressure, and vibration.

This project uses historical machine behaviour together with current sensor readings to provide multiple machine-health insights from a single input.

The system answers four main questions:

- Will the machine fail in the next 24 hours?
- Is the current machine behaviour abnormal?
- Why did the model produce this failure-risk prediction?
- Approximately how long until the next recorded machine failure?

The final application combines machine learning models, explainability, a FastAPI backend, and a React dashboard into one deployed system.

---

## Main Features

### Failure Prediction

Predicts whether a machine is at risk of failure within the next 24 hours.

Model:

`XGBoost Classifier`

The model uses current sensor readings together with historical and maintenance-related features.

Final decision threshold:

`0.60`

The interface reports this output as:

`Model-estimated Failure Risk`

---

### Anomaly Detection

Detects whether the current sensor behaviour is unusual compared with historical operating behaviour.

Model:

`Isolation Forest`

The anomaly detector is unsupervised and is kept separate from the supervised failure prediction model.

Output:

- Normal
- Anomaly
- Anomaly score

The anomaly score is not treated as a probability.

---

### SHAP Explainability

SHAP is used to explain the XGBoost failure prediction.

For every prediction, the system shows:

- Top features increasing failure risk
- Top features decreasing failure risk

Examples of contextual features include:

- Hours since last maintenance
- Errors in the last 24 hours
- 24-hour average vibration
- 24-hour voltage variation
- 6-hour sensor changes

SHAP values represent contributions to the model's raw log-odds output and are not causal effects or percentages.

---

### Remaining Useful Life Estimation

The project also includes an experimental RUL model that estimates:

`Time until the next recorded machine failure`

Model:

`XGBoost Regressor`

The RUL system uses both current readings and historical contextual features.

Final held-out test results:

| Metric | Result |
|---|---:|
| MAE | 593.55 hours |
| RMSE | 745.58 hours |
| R² | -0.0227 |

The RUL estimate is experimental.

The low test R² indicates that exact time-to-next-failure prediction is difficult using the available dataset and engineered features. Therefore, the RUL value should not be interpreted as a precise physical remaining lifetime.

---

## Machine Learning Pipeline

```text
Machine ID + Current Sensor Readings
                |
                v
      Retrieve Machine History
                |
                v
        Feature Engineering
                |
        -------------------
        |        |        |
        v        v        v
     XGBoost  Isolation  XGBoost
     Failure   Forest      RUL
      Model    Anomaly    Model
        |
        v
       SHAP
   Explanation
        |
        v
   Unified Machine
    Health Report

Feature Engineering
The user only needs to provide:
- Machine ID
- Voltage
- Rotation
- Pressure
- Vibration
The backend automatically derives contextual features such as:
- 24-hour rolling means
- 24-hour rolling standard deviations
- Current value vs 24-hour mean
- 6-hour sensor changes
- Errors in the last 24 hours
- Hours since last maintenance
- Machine age
The current reading is combined with recent machine history before prediction.
Dataset
The project uses the Microsoft/Azure Predictive Maintenance dataset.
Main data sources include:
- Telemetry
- Errors
- Maintenance
- Failures
- Machine information
Telemetry contains hourly readings for:
- Voltage
- Rotation
- Pressure
- Vibration
The original project uses approximately 876,000 telemetry records across 100 machines.
For deployment, a compact runtime history is used so the backend does not need to load the complete training dataset.
Model Evaluation
Failure Prediction
The final XGBoost failure model achieved approximately:
Metric	Result
Precision	94.8%
Recall	97.6%
F1 Score	96.2%
PR-AUC	0.9836


Machine-wise train, validation, and test splits were used so readings from the same machine were not randomly distributed across datasets.
Tech Stack
Machine Learning
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Imbalanced-learn
Backend
- FastAPI
- Uvicorn
- Pydantic
Frontend
- React
- Vite
- CSS
Deployment
- Render
- Netlify
- GitHub
API
Main prediction endpoint:
POST /predict

Example request:
{
  "machine_id": 1,
  "volt": 170.0,
  "rotate": 450.0,
  "pressure": 100.0,
  "vibration": 40.0
}

The API returns:
- Failure risk
- Failure prediction
- Anomaly status
- Anomaly score
- Estimated RUL
- SHAP explanation
Project Structure
Predictive-Maintenance/
│
├── backend/
│   ├── main.py
│   ├── inference.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── ...
│
├── models/
│   ├── failure_xgb_model.pkl
│   ├── anomaly_isolation_forest.pkl
│   ├── rul_xgb_model.pkl
│   └── ...
│
├── data/
│   └── runtime machine history
│
├── notebooks/
│   └── model.ipynb
│
└── README.md

Running Locally
Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

Backend:
http://127.0.0.1:8000

Swagger:
http://127.0.0.1:8000/docs

Frontend
cd frontend
npm install
npm run dev

Frontend:
http://localhost:5173

Future Improvements
- Sequence-based RUL prediction using LSTM or other deep-learning models
- Real-time streaming sensor data
- Continuous machine monitoring
- Maintenance alerts and notifications
- Improved RUL modelling using degradation-specific datasets
- Database integration
- Fleet-level machine monitoring dashboard
Author
Daksh Pandey
B.Tech Computer Science Engineering
Machine Learning / AI Project
