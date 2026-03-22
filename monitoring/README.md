# Monitoring Stack (Phase 11)

This stack provides advanced production monitoring for model serving with:

- Prometheus for scraping API metrics
- Grafana for visualization dashboards
- Online drift scoring against training-data baselines

## Metrics exposed by API

- `mlops_api_requests_total`
- `mlops_api_request_latency_seconds`
- `mlops_predictions_total`
- `mlops_prediction_score`
- `mlops_data_drift_score`
- `mlops_data_drift_alerts_total`
- `mlops_model_loaded`

Create secrets file first:

```bash
cp monitoring/env.monitoring.example .env.monitoring
```

Then edit `.env.monitoring` with your real Slack webhook + SMTP credentials.

```bash
docker compose up --build
```

If already running, reload Prometheus config/rules:

```bash
docker compose restart prometheus
```

## Access

- API docs: http://localhost:8000/docs
- API metrics: http://localhost:8000/metrics
- Prometheus: http://localhost:9090
- Alertmanager: http://localhost:9093
- Grafana: http://localhost:3000 (admin/admin)

Dashboard auto-loads as: **MLOps API Monitoring**

## Alert rules included

- `MLOpsAPIHighLatencyP95` (warning): p95 latency > 1s for 2m
- `MLOpsAPIErrorRateHigh` (critical): non-2xx ratio > 5% for 2m
- `MLOpsModelUnavailable` (critical): any model not loaded for 1m
- `MLOpsDataDriftWarning` (warning): drift score >= 2.0 for 1m
- `MLOpsDataDriftCritical` (critical): drift score >= 4.0 for 1m

## Quick test flow

1) Generate normal traffic (Swagger or curl/Postman)

- POST `http://localhost:8000/predict/spam`
- POST `http://localhost:8000/predict/cluster`
- POST `http://localhost:8000/predict/fraud`

2) Verify metrics in Prometheus (`/graph`)

Use these queries:

- `sum(rate(mlops_api_requests_total[1m])) by (endpoint)`
- `histogram_quantile(0.95, sum(rate(mlops_api_request_latency_seconds_bucket[5m])) by (le, endpoint))`
- `sum(increase(mlops_predictions_total[15m])) by (model, prediction)`
- `mlops_data_drift_score`
- `mlops_model_loaded`

3) Check alert state in Prometheus

- Open `http://localhost:9090/alerts`
- You should see all alert rules listed and current state (`inactive`/`pending`/`firing`)

## How to force-test alerts

### Data drift alert

Send extreme values to spike drift score:

```bash
curl -X POST "http://localhost:8000/predict/cluster" \
  -H "Content-Type: application/json" \
  -d "{\"features\":[99999,99999,99999,99999]}"
```

Then check:
- query `mlops_data_drift_score`
- alert page for `MLOpsDataDriftWarning` / `MLOpsDataDriftCritical`

### Error-rate alert

Call endpoint with invalid shape repeatedly to produce 4xx/5xx style failures:

```bash
curl -X POST "http://localhost:8000/predict/fraud" \
  -H "Content-Type: application/json" \
  -d "{\"features\":[1,2]}"
```

### Model-unavailable alert

Start API before models are trained/registered (or remove model source), then query:

- `mlops_model_loaded`

and check `MLOpsModelUnavailable`.

## Notification pipeline checks

1) Prometheus -> Alertmanager

- Open `http://localhost:9090/alerts`
- Any alert in `firing` should appear in Alertmanager UI `http://localhost:9093/#/alerts`

2) Alertmanager -> Slack

- Ensure `SLACK_WEBHOOK_URL` and `SLACK_CHANNEL` are valid in `.env.monitoring`
- Trigger drift alert using extreme payload
- Verify message arrives in Slack channel

3) Alertmanager -> Email (critical alerts)

- Ensure SMTP values are valid in `.env.monitoring`
- Trigger a critical alert (e.g., sustained high error rate)
- Verify email reaches `ALERT_EMAIL_TO`

## Manual alert injection test

Use Alertmanager API to validate delivery pipeline quickly:

```bash
curl -H "Content-Type: application/json" -d "[{\"labels\":{\"alertname\":\"ManualProductionTest\",\"severity\":\"critical\",\"service\":\"mlops-api\"},\"annotations\":{\"description\":\"Manual test alert for Slack/Email routing\"}}]" http://localhost:9093/api/v2/alerts
```
