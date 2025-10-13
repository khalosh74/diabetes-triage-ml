# api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from pathlib import Path
import joblib
import numpy as np

ARTIFACT_PATH = Path("artifacts/model.pkl")
if not ARTIFACT_PATH.exists():
    raise RuntimeError("Model artifact not found. Run training first.")

bundle = joblib.load(ARTIFACT_PATH)
pipe = bundle["pipeline"]
feature_order = bundle["feature_order"]

app = FastAPI(title="Virtual Diabetes Clinic Risk API", version=bundle["metadata"]["version"])

class PredictRequest(BaseModel):
    records: list[dict]

class PredictVectorRequest(BaseModel):
    vectors: list[list[float]]

    @field_validator("vectors")
    @classmethod
    def check_lengths(cls, v):
        n = len(feature_order)
        for i, vec in enumerate(v):
            if len(vec) != n:
                raise ValueError(f"vector {i} length={len(vec)}; expected {n}")
        return v

@app.get("/health")
def health():
    return {"status": "ok", "model": bundle["metadata"]}

@app.post("/predict")
def predict(req: PredictRequest):
    rows = []
    for rec in req.records:
        row = [rec.get(col) for col in feature_order]
        missing = [c for c, val in zip(feature_order, row) if val is None]
        if missing:
            raise HTTPException(status_code=400, detail={"error": "Missing features", "missing": missing})
        rows.append(row)
    preds = pipe.predict(np.array(rows))
    return {"scores": preds.tolist(), "feature_order": feature_order}

@app.post("/predict_vectors")
def predict_vectors(req: PredictVectorRequest):
    preds = pipe.predict(np.array(req.vectors))
    return {"scores": preds.tolist(), "feature_order": feature_order}


from fastapi.responses import RedirectResponse, HTMLResponse

@app.get("/", include_in_schema=False)
def root():
    # redirect to Swagger UI
    return RedirectResponse(url="/docs")
