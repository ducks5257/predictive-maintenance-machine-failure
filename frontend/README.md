# Predictive Maintenance Intelligence

React + Vite dashboard using the existing FastAPI backend. No model results are
hardcoded; **Load Sample** fills inputs only.

## Run locally

Start the existing backend on `http://127.0.0.1:8000`. Then, from this folder:

```bash
npm install
npm run dev
```

Open **http://localhost:5173**. Use **Load Sample**, then **Analyze Machine**.
Use `localhost` for the frontend address: it matches the backend's existing CORS
configuration. The dev server uses a strict port so it cannot silently move to
a different origin.

To change the API address, copy `.env.example` to `.env`, edit
`VITE_API_BASE_URL`, and restart Vite. A deployed frontend origin also needs to
be allowed by the backend's CORS configuration.

```bash
npm run build
npm run preview
```

Preview uses the same localhost port, so stop the dev server before previewing.

## Display conventions

- Failure risk is multiplied by 100 for display; the prediction and threshold
  come from the API.
- Anomaly score is a relative score, never a percentage.
- RUL displays the backend's estimated hours/days and keeps its warning visible.
- SHAP bars compare contribution magnitudes. Displayed values remain raw
  log-odds contributions, with no probability or causal interpretation.
