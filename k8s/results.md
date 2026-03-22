Perfect — now your stack is healthy.  
Use this exact **5-step demo script** for `roadmap.md (595-602)`:

## Demo Script

1) **Healthy stack**
```bash
kubectl get pods,svc,ingress -n mlops
kubectl get pvc -n mlops
curl http://localhost:8000/health
```

2) **Generate traffic**
```bash
for i in {1..50}; do
  curl -s -X POST "http://localhost:8000/predict/cluster" \
    -H "Content-Type: application/json" \
    -d "{\"features\":[25,50000,2,1]}" >/dev/null
done
echo "traffic-generated"
```

3) **Show metrics/alerts**
- Open Prometheus: `http://localhost:9090`
  - Query: `rate(mlops_api_requests_total[1m])`
  - Query: `histogram_quantile(0.95, sum(rate(mlops_api_request_latency_seconds_bucket[5m])) by (le, endpoint))`
- Open Grafana: `http://localhost:3000`
  - Show request rate + latency panels filling with data
- Optional alert test:
```bash
kubectl get --raw /apis/metrics.k8s.io/ >/dev/null 2>&1 || echo "metrics-server optional for full HPA demo"
```

4) **Pod failure auto-recovery**
```bash
kubectl delete pod -n mlops -l app=mlops-api
kubectl get pods -n mlops -w
```
(Show new API pod comes back automatically)

5) **Retraining job trigger**
```bash
kubectl create job --from=cronjob/retraining-cronjob retrain-manual -n mlops
kubectl get jobs -n mlops
kubectl logs -n mlops job/retrain-manual --tail=200
```

---

If you want, I can give you a **2-minute interview narration** for these same 5 steps (what to say while running each command).




histogram_quantile(0.95, sum(rate(mlops_api_request_latency_seconds_bucket[5m])) by (le, endpoint))
Evaluation time
Load time: 31ms   Result series: 1

{endpoint="/metrics"}	0.005083333333333337

rate(mlops_api_requests_total[1m])
Evaluation time
Load time: 26ms   Result series: 1

{endpoint="/metrics", http_status="200", instance="api:8000", job="mlops-api", method="GET"}
jects/mlops-infrastructure (clean-main2)
$ curl -v http://127.0.0.1:8000/health
*   Trying 127.0.0.1:8000...
* Connected to 127.0.0.1 (127.0.0.1) port 8000
* using HTTP/1.x
> GET /health HTTP/1.1
> Host: 127.0.0.1:8000
> User-Agent: curl/8.12.1
> Accept: */*
>
* Request completely sent off
< HTTP/1.1 200 OK
< date: Fri, 20 Mar 2026 20:00:49 GMT
< server: uvicorn
< content-length: 130
< content-type: application/json
<
{"status":"healthy","models_loaded":{"logistic_regression":false,"kmeans":false,
"isolation_forest":false},"mlflow_connected":true}* Connection #0 to host 127.0.
0.1 left intact
(venv)
PMLS@DESKTOP-OGUE2LK MINGW64 ~/Desktop/ARTIFICIAL INTELLIGENCE/MachineLearningProjects/mlops-infrastructure (clean-main2)
$ for i in {1..50}; do
  curl -s -X POST "http://localhost:8000/predict/cluster" \
    -H "Content-Type: application/json" \
    -d "{\"features\":[25,50000,2,1]}" >/dev/null
done
echo "traffic-generated"
traffic-generated
(venv)



Edit

Export 

Share



Last 30 minutes



Refresh

5s
API Request Rate by Endpoint
/metrics
/predict/cluster
Latency p95 by Endpoint
/metrics p95
/predict/cluster p95
Prediction Distribution (Last 15m)
Name
kmeans -> 2
Current Drift Score
12513
Drift Alerts (Last 15m)
kmeans critical
Model Availability

0
1
1