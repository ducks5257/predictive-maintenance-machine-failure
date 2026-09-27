import { useEffect, useState } from "react";
import { analyzeMachine, checkBackend } from "./api.js";

const emptyInputs = {
  machine_id: "",
  volt: "",
  rotate: "",
  pressure: "",
  vibration: "",
};
const sampleInputs = {
  machine_id: "1",
  volt: "170",
  rotate: "450",
  pressure: "100",
  vibration: "40",
};

const inputFields = [
  {
    name: "machine_id",
    label: "Machine ID",
    placeholder: "01",
    icon: "machine",
  },
  { name: "volt", label: "Voltage", placeholder: "170", icon: "bolt" },
  { name: "rotate", label: "Rotation", placeholder: "450", icon: "rotation" },
  { name: "pressure", label: "Pressure", placeholder: "100", icon: "gauge" },
  {
    name: "vibration",
    label: "Vibration",
    placeholder: "40",
    icon: "activity",
  },
];

const featureLabels = {
  volt: "Voltage",
  rotate: "Rotation",
  pressure: "Pressure",
  vibration: "Vibration",
  age: "Machine Age",
  errors_last_24h: "Errors in Last 24h",
  hours_since_last_maintenance: "Hours Since Last Maintenance",
  volt_24h_mean: "24h Average Voltage",
  rotate_24h_mean: "24h Average Rotation",
  pressure_24h_mean: "24h Average Pressure",
  vibration_24h_mean: "24h Average Vibration",
  volt_24h_std: "24h Voltage Variability",
  rotate_24h_std: "24h Rotation Variability",
  pressure_24h_std: "24h Pressure Variability",
  vibration_24h_std: "24h Vibration Variability",
  volt_vs_24h_mean: "Voltage vs. 24h Average",
  rotate_vs_24h_mean: "Rotation vs. 24h Average",
  pressure_vs_24h_mean: "Pressure vs. 24h Average",
  vibration_vs_24h_mean: "Vibration vs. 24h Average",
  volt_6h_change: "6h Voltage Change",
  rotate_6h_change: "6h Rotation Change",
  pressure_6h_change: "6h Pressure Change",
  vibration_6h_change: "6h Vibration Change",
};

const defaultRulWarning =
  "Experimental estimate. Final R² was low, so this should not be treated as a precise physical remaining lifetime.";

// Small inline icons keep the interface independent of a UI or icon library.
function Icon({ name, size = 20, className = "" }) {
  const shapes = {
    activity: <path d="M2 12h4l3-8 5 16 3-8h5" />,
    machine: (
      <>
        <rect x="4" y="4" width="16" height="16" rx="3" />
        <path d="M8 8h8M8 12h3m-3 4h8" />
      </>
    ),
    bolt: <path d="m13 2-9 12h7l-1 8 10-13h-8z" />,
    rotation: (
      <>
        <path d="M20 11a8 8 0 1 0-2 6M20 4v7h-7" />
        <circle cx="12" cy="12" r="2" />
      </>
    ),
    gauge: (
      <>
        <path d="M4 19a10 10 0 1 1 16 0H4Z" />
        <path d="m12 13 4-5M6 13h1m5-8v1m5 7h1" />
        <circle cx="12" cy="13" r="1" />
      </>
    ),
    arrow: <path d="M4 12h16m-6-6 6 6-6 6" />,
    sample: (
      <>
        <rect x="5" y="4" width="14" height="17" rx="2" />
        <path d="M9 2h6v4H9zM9 11h6m-6 5h4" />
      </>
    ),
    shield: (
      <>
        <path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6z" />
        <path d="m8 12 3 3 5-6" />
      </>
    ),
    clock: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 2" />
      </>
    ),
    layers: (
      <>
        <path d="m12 3 10 5-10 5L2 8zM2 12l10 5 10-5M2 16l10 5 10-5" />
      </>
    ),
    info: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 11v6m0-10v.1" />
      </>
    ),
    up: <path d="M12 19V5m-6 6 6-6 6 6" />,
    down: <path d="M12 5v14m-6-6 6 6 6-6" />,
    check: <path d="m5 12 4 4L19 6" />,
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={className}
    >
      {shapes[name]}
    </svg>
  );
}

function formatNumber(value, digits = 1) {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits,
  }).format(value);
}

function FactorList({ items, direction, maximum }) {
  if (items.length === 0) {
    return (
      <p className="no-factors">
        No {direction === "up" ? "increasing" : "decreasing"} contributions for
        this reading.
      </p>
    );
  }
  return (
    <ol className={`factor-list factors-${direction}`}>
      {items.slice(0, 3).map((item, index) => (
        <li className="factor" key={item.feature}>
          <span className="factor-number">0{index + 1}</span>
          <div className="factor-body">
            <div className="factor-details">
              <span className="factor-name" title={item.feature}>
                {featureLabels[item.feature] ||
                  item.feature.replaceAll("_", " ")}
              </span>
              <span className="factor-value">
                {item.shap_value > 0 ? "+" : ""}
                {item.shap_value.toFixed(4)}
              </span>
            </div>
            <div className="factor-track" aria-hidden="true">
              <span
                style={{
                  width: `${maximum > 0 ? (Math.abs(item.shap_value) / maximum) * 100 : 0}%`,
                }}
              />
            </div>
          </div>
        </li>
      ))}
    </ol>
  );
}

export default function App() {
  const [inputs, setInputs] = useState(emptyInputs);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [isOnline, setIsOnline] = useState(null);

  useEffect(() => {
    let active = true;
    async function refreshStatus() {
      const online = await checkBackend();
      if (active) setIsOnline(online);
    }
    refreshStatus();
    const interval = setInterval(refreshStatus, 30000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  function changeInput(event) {
    setInputs({ ...inputs, [event.target.name]: event.target.value });
    // Clear old results so they cannot be mistaken for the edited readings.
    setResult(null);
    setError("");
  }

  function loadSample() {
    setInputs({ ...sampleInputs });
    setResult(null);
    setError("");
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (isLoading) return;
    const reading = {
      machine_id: Number(inputs.machine_id),
      volt: Number(inputs.volt),
      rotate: Number(inputs.rotate),
      pressure: Number(inputs.pressure),
      vibration: Number(inputs.vibration),
    };
    if (!Number.isInteger(reading.machine_id) || reading.machine_id < 1) {
      setError("Enter a valid whole-number machine ID.");
      return;
    }
    for (const value of Object.values(reading)) {
      if (!Number.isFinite(value)) {
        setError("Enter a valid number in every sensor field.");
        return;
      }
    }

    setIsLoading(true);
    setError("");
    setResult(null);
    try {
      const prediction = await analyzeMachine(reading);
      setResult(prediction);
      setIsOnline(true);
    } catch (requestError) {
      setError(requestError.message);
      setIsOnline(await checkBackend());
    } finally {
      setIsLoading(false);
    }
  }

  const riskPercent = result ? result.failure.risk * 100 : null;
  const thresholdPercent = result ? result.failure.threshold * 100 : null;
  const highRisk = result?.failure.prediction === "Yes";
  const anomalous = result?.anomaly.status === "Anomaly";
  const increasing = result?.explanation.risk_increasing_features || [];
  const decreasing = result?.explanation.risk_decreasing_features || [];
  let largestContribution = 0;
  for (const factor of [...increasing, ...decreasing]) {
    largestContribution = Math.max(
      largestContribution,
      Math.abs(factor.shap_value),
    );
  }
  let connectionLabel = "Connecting";
  if (isOnline === true) connectionLabel = "System Online";
  if (isOnline === false) connectionLabel = "Backend Offline";

  return (
    <div className="app-shell">
      <header className="topbar">
        <a
          className="brand"
          href="#main-content"
          aria-label="Predictive Maintenance home"
        >
          <span className="brand-mark">
            <Icon name="activity" size={23} />
          </span>
          <span>
            PREDICTIVE<span className="brand-secondary"> / INTELLIGENCE</span>
          </span>
        </a>
        <div
          className={`connection-badge ${isOnline === true ? "online" : isOnline === false ? "offline" : "connecting"}`}
          role="status"
        >
          <span className="status-dot" />
          {connectionLabel}
        </div>
      </header>

      <main id="main-content" className="dashboard">
        <section className="intro" aria-labelledby="page-title">
          <div>
            <p className="eyebrow">
              <span /> MACHINE DIAGNOSTICS
            </p>
            <h1 id="page-title">
              Predictive Maintenance <span>Intelligence</span>
            </h1>
            <p className="intro-description">
              Real-time machine health analysis powered by machine learning
            </p>
          </div>
          <div className="intro-detail" aria-hidden="true">
            <Icon name="layers" size={27} />
            <span>
              Understand the signals.
              <br />
              <strong>Plan the next move.</strong>
            </span>
          </div>
        </section>

        <section className="panel input-panel" aria-labelledby="input-title">
          <form onSubmit={handleSubmit}>
            <div className="panel-heading">
              <div className="heading-with-icon">
                <span className="section-icon">
                  <Icon name="machine" />
                </span>
                <div>
                  <h2 id="input-title">Machine input</h2>
                  <p>
                    Enter current readings. Historical context is added
                    automatically.
                  </p>
                </div>
              </div>
              <span className="quiet-tag">NEXT-HOUR ANALYSIS</span>
            </div>
            <fieldset
              className="input-grid"
              disabled={isLoading}
              aria-labelledby="input-title"
            >
              {inputFields.map((field) => (
                <div
                  className={`input-field ${field.name === "machine_id" ? "machine-field" : ""}`}
                  key={field.name}
                >
                  <label htmlFor={field.name}>
                    <Icon name={field.icon} size={15} />
                    {field.label}
                  </label>
                  <input
                    id={field.name}
                    name={field.name}
                    type="number"
                    required
                    step={field.name === "machine_id" ? "1" : "any"}
                    min={field.name === "machine_id" ? "1" : undefined}
                    placeholder={field.placeholder}
                    value={inputs[field.name]}
                    onChange={changeInput}
                    autoComplete="off"
                  />
                </div>
              ))}
            </fieldset>
            <div className="input-footer">
              <p>
                <Icon name="info" size={15} />
                Uses the selected machine’s stored history.
              </p>
              <div className="form-actions">
                <button
                  className="button button-secondary"
                  type="button"
                  onClick={loadSample}
                  disabled={isLoading}
                >
                  <Icon name="sample" size={17} />
                  Load Sample
                </button>
                <button
                  className="button button-primary"
                  type="submit"
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <>
                      <span className="spinner" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      Analyze Machine
                      <Icon name="arrow" size={18} />
                    </>
                  )}
                </button>
              </div>
            </div>
          </form>
        </section>

        {error && (
          <div className="error-card" role="alert">
            <Icon name="info" size={22} />
            <div>
              <strong>Analysis unavailable</strong>
              <p>{error}</p>
            </div>
          </div>
        )}
        <p className="sr-only" role="status" aria-live="polite">
          {isLoading
            ? "Analyzing machine readings."
            : result
              ? `Analysis complete for machine ${result.machine_id}.`
              : ""}
        </p>

        <section
          className={`results-section ${result ? "has-result" : ""} ${isLoading ? "is-loading" : ""}`}
          aria-labelledby="results-title"
          aria-busy={isLoading}
        >
          <div className="results-heading">
            <div>
              <span className="section-number">01</span>
              <h2 id="results-title">Machine insights</h2>
            </div>
            <span className={`results-state ${result ? "ready" : ""}`}>
              {result ? (
                <>
                  <Icon name="check" size={14} />
                  Machine {String(result.machine_id).padStart(2, "0")} ·
                  Analysis complete
                </>
              ) : isLoading ? (
                "Reading the signals..."
              ) : (
                "Awaiting analysis"
              )}
            </span>
          </div>

          <div className="metrics-grid">
            <article
              className={`panel metric-card failure-card ${highRisk ? "attention" : ""}`}
              aria-labelledby="failure-title"
            >
              <div className="metric-heading">
                <span className="metric-icon">
                  <Icon name="shield" />
                </span>
                <h3 id="failure-title">Failure Risk</h3>
                <span className="card-index">01</span>
              </div>
              <div className="risk-value">
                <span>
                  {riskPercent === null
                    ? "—"
                    : riskPercent > 0 && riskPercent < 0.0001
                      ? "<0.0001"
                      : formatNumber(riskPercent, 4)}
                </span>
                <small>%</small>
              </div>
              <p className="metric-caption">Model-estimated Failure Risk</p>
              <div
                className="risk-track"
                role={result ? "progressbar" : undefined}
                aria-label="Model-estimated failure risk"
                aria-valuemin={result ? 0 : undefined}
                aria-valuemax={result ? 100 : undefined}
                aria-valuenow={riskPercent ?? undefined}
              >
                <span
                  className="risk-fill"
                  style={{ width: `${riskPercent ?? 0}%` }}
                />
                {result && (
                  <span
                    className="threshold-marker"
                    style={{ left: `${thresholdPercent}%` }}
                    title={`Threshold: ${formatNumber(thresholdPercent)}%`}
                  />
                )}
              </div>
              <div className="risk-scale">
                <span>0%</span>
                <span>
                  Threshold{" "}
                  {result ? `${formatNumber(thresholdPercent)}%` : "—"}
                </span>
                <span>100%</span>
              </div>
              <div className="metric-bottom">
                <span>Failure prediction</span>
                <span
                  className={`result-pill ${result ? (highRisk ? "warning" : "positive") : "neutral"}`}
                >
                  {result ? (
                    <>
                      <span className="status-dot" />
                      {result.failure.prediction}
                    </>
                  ) : (
                    "No result yet"
                  )}
                </span>
              </div>
            </article>

            <article
              className={`panel metric-card anomaly-card ${anomalous ? "attention" : ""}`}
              aria-labelledby="anomaly-title"
            >
              <div className="metric-heading">
                <span className="metric-icon">
                  <Icon name="activity" />
                </span>
                <h3 id="anomaly-title">Anomaly Detection</h3>
                <span className="card-index">02</span>
              </div>
              <div
                className={`anomaly-value ${result ? (anomalous ? "warning-text" : "positive-text") : ""}`}
              >
                {result ? (
                  <>
                    <span className="status-dot" />
                    {result.anomaly.status}
                  </>
                ) : (
                  "—"
                )}
              </div>
              <p className="metric-caption">Operating pattern assessment</p>
              <div className="score-row">
                <span>Anomaly score</span>
                <strong>
                  {result ? formatNumber(result.anomaly.score, 4) : "—"}
                </strong>
              </div>
              <p className="anomaly-help">
                Measures whether current sensor behaviour differs from learned
                operating patterns.
              </p>
              <div className="metric-bottom">
                <span>Relative score · not a probability</span>
              </div>
            </article>

            <article
              className="panel metric-card rul-card"
              aria-labelledby="rul-title"
            >
              <div className="metric-heading">
                <span className="metric-icon">
                  <Icon name="clock" />
                </span>
                <h3 id="rul-title">Estimated Remaining Useful Life</h3>
                <span className="card-index">03</span>
              </div>
              <div className="rul-value">
                {result ? formatNumber(result.rul.estimated_hours) : "—"}
                <small>hours</small>
              </div>
              <div className="rul-days">
                {result
                  ? `≈ ${formatNumber(result.rul.estimated_days)} days`
                  : "Waiting for an estimate"}
              </div>
              <p className="rul-label">
                Model-estimated time to next recorded failure
              </p>
              {result && result.rul.raw_hours < 0 && (
                <p className="raw-rul-note">
                  Raw output: {formatNumber(result.rul.raw_hours)} hours.
                  Display limited to zero.
                </p>
              )}
              <div className="rul-warning">
                <Icon name="info" size={16} />
                <p>{result?.rul.note || defaultRulWarning}</p>
              </div>
            </article>
          </div>
        </section>

        <section
          className={`panel explanation-panel ${result ? "has-result" : ""}`}
          aria-labelledby="explanation-title"
        >
          <div className="panel-heading">
            <div className="heading-with-icon">
              <span className="section-icon">
                <Icon name="layers" />
              </span>
              <div>
                <h2 id="explanation-title">
                  Why did the model make this prediction?
                </h2>
                <p>The strongest contributions to this failure estimate.</p>
              </div>
            </div>
            <span className="quiet-tag">LOCAL EXPLANATION</span>
          </div>
          {result ? (
            <div className="factors-grid">
              <div className="factors-column">
                <h3>
                  <span className="direction-icon increasing">
                    <Icon name="up" size={15} />
                  </span>
                  Risk Increasing Factors
                </h3>
                <FactorList
                  items={increasing}
                  direction="up"
                  maximum={largestContribution}
                />
              </div>
              <div className="factors-column">
                <h3>
                  <span className="direction-icon decreasing">
                    <Icon name="down" size={15} />
                  </span>
                  Risk Decreasing Factors
                </h3>
                <FactorList
                  items={decreasing}
                  direction="down"
                  maximum={largestContribution}
                />
              </div>
            </div>
          ) : (
            <div className="explanation-empty">
              <div className="empty-orbit">
                <Icon name="layers" size={25} />
              </div>
              <div>
                <h3>
                  {isLoading
                    ? "Connecting readings to insights"
                    : "Every prediction has a story"}
                </h3>
                <p>
                  {isLoading
                    ? "Preparing the local model explanation..."
                    : "Analyze a machine to discover which features influence its failure estimate."}
                </p>
              </div>
            </div>
          )}
          <p className="explanation-note">
            <Icon name="info" size={14} />
            SHAP values are model contributions in raw log-odds space, not
            percentages or causal effects.
          </p>
        </section>

        <footer className="page-footer">
          <span>
            <span className="footer-mark" />
            PREDICTIVE MAINTENANCE INTELLIGENCE
          </span>
          <p>Machine learning insights. Informed human decisions.</p>
        </footer>
      </main>
    </div>
  );
}
