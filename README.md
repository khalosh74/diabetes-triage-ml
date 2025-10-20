# Virtual Diabetes Clinic — ML Triage Service

Predict a **short‑term progression risk score** for diabetes check‑ins (higher = worse). v0.2.0 also supports an optional **high‑risk flag** using a learned threshold. Everything is reproducible, Dockerized, and automated with GitHub Actions.

---

## What this repo contains

- **API**: FastAPI service with Swagger at `/docs`
- **Model**:
  - **v0.1 (baseline)** – `StandardScaler → LinearRegression`
  - **v0.2.0 (improved)** – Trains **Ridge** and **RandomForestRegressor**, selects the best RMSE, adds a **high‑risk** flag
- **Artifacts**: `artifacts/model.pkl`, `artifacts/metrics.json` (created by `train.py` or during Docker build)
- **Pipelines**: GitHub Actions for CI and Release
- **Reproducibility**: pinned deps, fixed seeds, deterministic training in Docker build

---

## Run it (Docker – recommended, no Python needed)

```bash
# from repo root
docker build -t diaclinic:0.2.0 .
docker run --rm -p 8000:8000 diaclinic:0.2.0
```

Open: http://localhost:8000/ → redirects to `/docs`

Port busy? Use another host port: `docker run --rm -p 9000:8000 diaclinic:0.2.0` then visit http://localhost:9000/.

---

## Run it (local Python – optional)

```bash
python3 -m venv .venv
# mac/linux
source .venv/bin/activate
# windows powershell: .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt

# Train (writes artifacts/)
python src/diaclinic/train.py

# Serve
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## API reference (most used)

### Endpoints
- `GET /health` → `{"status":"ok","model_version":"<semver>", ...}`
- `GET /ready`  → readiness probe
- `GET /metadata` → model name/version, `feature_order`, learned `high_risk_threshold`
- `POST /predict_one` → one record → one score
- `POST /predict` → list of objects (10 features each) → scores (+ optional flags)
- `POST /predict_vectors` → list of arrays (10 floats in exact order) → scores (+ optional flags)

### Features (always 10)
```
age, sex, bmi, bp, s1, s2, s3, s4, s5, s6
```

### Example — `POST /predict_one`
**Request**
```json
{
  "age": 0.05, "sex": -0.04, "bmi": 0.03, "bp": 0.02,
  "s1": 0.01, "s2": 0.0, "s3": -0.01, "s4": 0.02, "s5": 0.03, "s6": -0.02
}
```
**Response**
```json
{ "prediction": 201.82 }
```

### Example — `POST /predict` (records + high‑risk flag)
**Request**
```json
{
  "records": [
    { "age": 0.05, "sex": -0.04, "bmi": 0.03, "bp": 0.02, "s1": 0.01, "s2": 0.0, "s3": -0.01, "s4": 0.02, "s5": 0.03, "s6": -0.02 }
  ],
  "include_flag": true
}
```
**Response**
```json
{
  "scores": [201.82],
  "feature_order": ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"],
  "flags": [0],
  "threshold_used": 214.0
}
```

> To override the learned threshold, include `"threshold": <number>` in the request.

---

## Reproduce training

- Dataset: `sklearn.datasets.load_diabetes(as_frame=True)`
- Split: `test_size=0.2`, `random_state=42`
- **v0.1**: `StandardScaler + LinearRegression` → report RMSE
- **v0.2.0**: Train **Ridge** & **RandomForestRegressor**, choose **best RMSE**; compute **high‑risk threshold** at the **0.75 quantile of y_train**; report **precision/recall @ threshold**

Artifacts saved to `artifacts/`:
- `model.pkl` – fitted model/pipeline + metadata (feature order, threshold, candidates)
- `metrics.json` – baseline, candidates, selected, and deltas vs baseline
- `CHANGELOG.md` – auto‑appended summary for v0.2.0

Training locally:
```bash
python src/diaclinic/train.py
cat artifacts/metrics.json
```

Training in Docker happens **during `docker build`**.

---

## CI/CD (GitHub Actions)

- **CI** (on push/PR): lint, test, train (smoke), upload `artifacts/`
- **Release** (on tag `v*`): train, build & push image to **GHCR**, container smoke test, create GitHub Release attaching `model.pkl` + `metrics.json`

Cut releases:
```bash
git tag -a v0.1.0 -m "Baseline"
git push origin v0.1.0

git tag -a v0.2.0 -m "Ridge/RF + high-risk flag"
git push origin v0.2.0
```

---

## Notes

- Everything is inside the Docker image (Python, deps, training, artifacts). No virtualenv required to run.
- Swagger examples in `/docs` show valid request bodies.
- CORS is open for local dev; restrict `ALLOW_ORIGINS` env when deploying.
- License: add your preferred license in `LICENSE`.




