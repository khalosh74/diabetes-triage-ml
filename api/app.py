from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, conlist
import joblib
from pathlib import Path
import numpy as np

ARTIFACT_PATH = Path("artifacts/model.pkl")
if not ARTIFACT_PATH.exists():
    raise RuntimeError("Model artifact not found. Run training first.")

bundle = joblib.load(ARTIFACT_PATH)
pipe = bundle["pipeline"]
feature_order = bundle["feature_order"]
meta = bundle["metadata"]
DEFAULT_THR = float(meta.get("high_risk_threshold"))

app = FastAPI(title="Virtual Diabetes Clinic Risk API", version=meta["version"])

class PredictRequest(BaseModel):
    records: list[dict]
    include_flag: bool | None = False
    # optional: override threshold per-request
    threshold: float | None = None

class PredictVectorRequest(BaseModel):
    vectors: list[conlist(float, min_items=len(feature_order), max_items=len(feature_order))]
    include_flag: bool | None = False
    threshold: float | None = None

@app.get("/health")
def health():
    return {"status": "ok", "model": meta}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        rows = []
        for rec in req.records:
            row = [rec.get(col, None) for col in feature_order]
            if any(v is None for v in row):
                missing = [c for c, v in zip(feature_order, row) if v is None]
                raise HTTPException(status_code=400, detail={"error": "Missing features", "missing": missing})
            rows.append(row)
        preds = pipe.predict(np.array(rows))
        payload = {"scores": preds.tolist(), "feature_order": feature_order}
        if req.include_flag:
            thr = req.threshold if req.threshold is not None else DEFAULT_THR
            flags = (preds >= thr).astype(int).tolist()
            payload.update({"flags": flags, "threshold_used": thr})
        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "prediction_failed", "message": str(e)})

@app.post("/predict_vectors")
def predict_vectors(req: PredictVectorRequest):
    try:
        preds = pipe.predict(np.array(req.vectors))
        payload = {"scores": preds.tolist(), "feature_order": feature_order}
        if req.include_flag:
            thr = req.threshold if req.threshold is not None else DEFAULT_THR
            flags = (preds >= thr).astype(int).tolist()
            payload.update({"flags": flags, "threshold_used": thr})
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "prediction_failed", "message": str(e)})
