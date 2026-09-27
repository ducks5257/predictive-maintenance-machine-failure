// Set VITE_API_BASE_URL in .env when the backend address changes.
export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

const connectionMessage =
  "Unable to analyze machine. Please verify that the backend is running and try again.";
const invalidResponseMessage =
  "The backend returned an incomplete or invalid result. Please try again.";

// Check the fields the dashboard displays before presenting any model output.
function isValidPrediction(data) {
  if (!data || !Number.isInteger(data.machine_id)) return false;
  const { failure, anomaly, rul, explanation } = data;
  if (!failure || !anomaly || !rul || !explanation) return false;

  const numbers = [
    failure.risk,
    failure.threshold,
    anomaly.score,
    rul.raw_hours,
    rul.estimated_hours,
    rul.estimated_days,
  ];
  for (const value of numbers) {
    if (!Number.isFinite(value)) return false;
  }
  if (failure.risk < 0 || failure.risk > 1) return false;
  if (failure.threshold < 0 || failure.threshold > 1) return false;
  if (!["Yes", "No"].includes(failure.prediction)) return false;
  if (!["Normal", "Anomaly"].includes(anomaly.status)) return false;
  if (rul.estimated_hours < 0 || rul.estimated_days < 0) return false;
  if (typeof rul.note !== "string" || !rul.note.trim()) return false;

  for (const items of [
    explanation.risk_increasing_features,
    explanation.risk_decreasing_features,
  ]) {
    if (!Array.isArray(items)) return false;
    for (const item of items) {
      if (!item || typeof item.feature !== "string") return false;
      if (!Number.isFinite(item.shap_value)) return false;
    }
  }
  return true;
}

export async function analyzeMachine(reading) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reading),
      signal: controller.signal,
    });

    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error(response.ok ? invalidResponseMessage : connectionMessage);
    }

    if (!response.ok) {
      if (response.status === 422) {
        throw new Error(
          "Check the machine ID and all four sensor readings, then try again.",
        );
      }
      if (
        response.status < 500 &&
        typeof data.detail === "string" &&
        data.detail.length < 200
      ) {
        throw new Error(data.detail);
      }
      throw new Error(connectionMessage);
    }

    if (!isValidPrediction(data) || data.machine_id !== reading.machine_id) {
      throw new Error(invalidResponseMessage);
    }
    return data;
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("Analysis took longer than expected. Please try again.");
    }
    if (error instanceof TypeError) throw new Error(connectionMessage);
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

export async function checkBackend() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!response.ok) return false;
    const data = await response.json();
    return data.status === "healthy";
  } catch {
    return false;
  }
}
