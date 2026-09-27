"""Load the frozen deployment objects once and predict from stored history."""

from operator import itemgetter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Module imports happen once per backend process, not once per request.
smote_xgb = joblib.load(MODELS_DIR / "failure_xgb_model.pkl")
feature_imputer = joblib.load(MODELS_DIR / "failure_imputer.pkl")
feature_columns = joblib.load(MODELS_DIR / "failure_feature_columns.pkl")
selected_smote_threshold = joblib.load(MODELS_DIR / "failure_threshold.pkl")

isolation_forest = joblib.load(MODELS_DIR / "anomaly_isolation_forest.pkl")
anomaly_imputer = joblib.load(MODELS_DIR / "anomaly_imputer.pkl")
anomaly_feature_columns = joblib.load(MODELS_DIR / "anomaly_feature_columns.pkl")

final_rul_xgb_model = joblib.load(MODELS_DIR / "rul_xgb_model.pkl")
final_rul_imputer = joblib.load(MODELS_DIR / "rul_imputer.pkl")
rul_context_features = joblib.load(MODELS_DIR / "rul_feature_columns.pkl")

shap_background = joblib.load(MODELS_DIR / "shap_background.pkl")
deployment_metadata = joblib.load(MODELS_DIR / "deployment_metadata.pkl")

# Live inference needs history and machine metadata, but no failure records.
telemetry = pd.read_csv(DATA_DIR / "PdM_telemetry.csv")
errors = pd.read_csv(DATA_DIR / "PdM_errors.csv")
maint = pd.read_csv(DATA_DIR / "PdM_maint.csv")
machines = pd.read_csv(DATA_DIR / "PdM_machines.csv")

for history_frame in [telemetry, errors, maint]:
    history_frame["datetime"] = pd.to_datetime(
        history_frame["datetime"], utc=True, errors="raise"
    )
    if history_frame["datetime"].isna().any():
        raise ValueError("Runtime history contains missing timestamps.")
    history_frame.sort_values(["machineID", "datetime"], inplace=True)

# Match the notebook's TreeExplainer configuration and keep all background rows.
failure_shap_explainer = shap.Explainer(
    smote_xgb,
    masker=shap.maskers.Independent(
        shap_background, max_samples=len(shap_background)
    ),
    algorithm="tree",
    feature_perturbation="interventional",
    model_output="raw",
    feature_names=feature_columns,
)


def build_machine_features(machine_id, volt, rotate, pressure, vibration):
    """Reproduce the notebook's next-hour features for one machine."""
    machine_metadata = machines.loc[machines["machineID"] == machine_id]
    if machine_metadata.empty:
        raise ValueError(f"Unknown machine ID: {machine_id}.")
    if len(machine_metadata) != 1:
        raise ValueError("The machine must have exactly one metadata record.")

    machine_history = telemetry.loc[telemetry["machineID"] == machine_id]
    if machine_history.empty:
        raise ValueError("The selected machine has no stored telemetry.")
    prediction_time = machine_history["datetime"].max() + pd.Timedelta(hours=1)
    if prediction_time != prediction_time.floor("h"):
        raise ValueError("Stored telemetry must use exact hourly timestamps.")

    sensors = ["volt", "rotate", "pressure", "vibration"]
    current = pd.Series(
        [volt, rotate, pressure, vibration], index=sensors, dtype="float64"
    )
    if not np.isfinite(current.to_numpy()).all():
        raise ValueError("All four sensor readings must be finite numbers.")

    # Preserve the notebook's protection against numerical overflow.
    numeric_limit = float(np.finfo(np.float32).max / 100)
    current = current.clip(-numeric_limit, numeric_limit)

    age = float(machine_metadata.iloc[0]["age"])
    if not np.isfinite(age):
        raise ValueError("The selected machine's age is missing or invalid.")

    # The 24-reading window includes the new reading and up to 23 earlier hours.
    history = machine_history.tail(23)
    expected_hours = pd.date_range(
        end=prediction_time - pd.Timedelta(hours=1), periods=len(history), freq="h"
    )
    if not pd.DatetimeIndex(history["datetime"]).equals(expected_hours):
        raise ValueError("The selected machine needs consecutive hourly history.")
    past_values = history[sensors].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(past_values.to_numpy()).all():
        raise ValueError("Stored sensor history contains missing or infinite values.")

    window_values = pd.concat([past_values, current.to_frame().T], ignore_index=True)
    features = current.to_dict()
    features["age"] = age
    for sensor in sensors:
        mean = window_values[sensor].mean()
        features[f"{sensor}_24h_mean"] = mean
        if len(window_values) > 1:
            features[f"{sensor}_24h_std"] = window_values[sensor].std(ddof=1)
        else:
            features[f"{sensor}_24h_std"] = 0.0
        features[f"{sensor}_vs_24h_mean"] = current[sensor] - mean
        if len(history) >= 6:
            features[f"{sensor}_6h_change"] = current[sensor] - past_values[sensor].iloc[-6]
        else:
            features[f"{sensor}_6h_change"] = 0.0

    # Match the notebook's boundaries: errors in (t-24h, t], maintenance <= t.
    recent_errors = errors.loc[
        (errors["machineID"] == machine_id)
        & (errors["datetime"] > prediction_time - pd.Timedelta(hours=24))
        & (errors["datetime"] <= prediction_time)
    ]
    machine_maintenance = maint.loc[
        (maint["machineID"] == machine_id)
        & (maint["datetime"] <= prediction_time)
    ]
    features["errors_last_24h"] = len(recent_errors)
    if machine_maintenance.empty:
        features["hours_since_last_maintenance"] = np.nan
    else:
        last_maintenance = machine_maintenance["datetime"].max()
        features["hours_since_last_maintenance"] = (
            prediction_time - last_maintenance
        ).total_seconds() / 3600

    return pd.DataFrame([features], columns=feature_columns)


def predict_machine_health(machine_id, volt, rotate, pressure, vibration):
    """Return separate failure, anomaly, RUL, and local SHAP outputs."""
    feature_row = build_machine_features(machine_id, volt, rotate, pressure, vibration)

    # Use exactly the same float32 failure input for prediction and SHAP.
    failure_input = pd.DataFrame(
        feature_imputer.transform(feature_row),
        columns=feature_columns,
        index=feature_row.index,
    ).astype("float32")
    failure_class_index = list(smote_xgb.classes_).index(1)
    failure_risk = float(smote_xgb.predict_proba(failure_input)[0, failure_class_index])
    failure_prediction = "Yes" if failure_risk >= selected_smote_threshold else "No"

    anomaly_features = feature_row.loc[:, anomaly_feature_columns]
    anomaly_features = anomaly_features.replace([np.inf, -np.inf], np.nan)
    anomaly_input = pd.DataFrame(
        anomaly_imputer.transform(anomaly_features),
        columns=anomaly_feature_columns,
        index=feature_row.index,
    )
    anomaly_score = float(-isolation_forest.decision_function(anomaly_input)[0])
    anomaly_status = "Anomaly" if anomaly_score > 0 else "Normal"

    rul_features = feature_row.loc[:, rul_context_features]
    rul_features = rul_features.replace([np.inf, -np.inf], np.nan)
    rul_input = pd.DataFrame(
        final_rul_imputer.transform(rul_features),
        columns=rul_context_features,
        index=feature_row.index,
    )
    raw_rul_hours = float(final_rul_xgb_model.predict(rul_input)[0])
    estimated_rul_hours = max(0.0, raw_rul_hours)
    estimated_rul_days = estimated_rul_hours / 24

    explanation = failure_shap_explainer(
        failure_input, check_additivity=True, approximate=False
    )
    shap_values = explanation.values[0]
    risk_increasing_features = []
    risk_decreasing_features = []
    for position, feature in enumerate(feature_columns):
        contribution = float(shap_values[position])
        item = {"feature": str(feature), "shap_value": contribution}
        if contribution > 0:
            risk_increasing_features.append(item)
        elif contribution < 0:
            risk_decreasing_features.append(item)

    risk_increasing_features = sorted(
        risk_increasing_features, key=itemgetter("shap_value"), reverse=True
    )[:3]
    risk_decreasing_features = sorted(
        risk_decreasing_features, key=itemgetter("shap_value")
    )[:3]

    return {
        "machine_id": int(machine_id),
        "failure": {
            "label": "Model-estimated Failure Risk",
            "risk": failure_risk,
            "threshold": float(selected_smote_threshold),
            "prediction": failure_prediction,
        },
        "anomaly": {"status": anomaly_status, "score": anomaly_score},
        "rul": {
            "raw_hours": raw_rul_hours,
            "estimated_hours": estimated_rul_hours,
            "estimated_days": estimated_rul_days,
            "label": "Model-estimated time to next recorded failure",
            "note": str(deployment_metadata["rul_warning"]),
        },
        "explanation": {
            "value_space": "failure model raw log-odds",
            "risk_increasing_features": risk_increasing_features,
            "risk_decreasing_features": risk_decreasing_features,
        },
    }
