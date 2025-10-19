from __future__ import annotations

import os
from pathlib import Path
from typing import List
import pandas as pd

import joblib
import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field, ConfigDict

# =========================
# Load artifacts at startup
# =========================
ARTIFACT_PATH = Path("artifacts/model.pkl")
if not ARTIFACT_PATH.exists():
    raise RuntimeError("Model artifact not found. Run training first.")

bundle = joblib.load(ARTIFACT_PATH)
pipe = bundle["pipeline"]
feature_order: List[str] = bundle["feature_order"]
meta: dict = bundle["metadata"]

# Guard: support older (v0.1) artifacts that have no threshold
thr_raw = meta.get("high_risk_threshold", None)
DEFAULT_THR: float | None = float(thr_raw) if thr_raw is not None else None

# ============
# FastAPI app
# ============
app = FastAPI(
    title="Virtual Diabetes Clinic Risk API",
    version=str(meta.get("version", "0.0.0")),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ---------------------------
# CORS (open for local dev)
# Use comma-separated list in ALLOW_ORIGINS to lock down in prod
# ---------------------------
allow_origins_env = os.getenv("ALLOW_ORIGINS", "*")
allow_origins = (
    ["*"]
    if allow_origins_env.strip() == "*"
    else [o.strip() for o in allow_origins_env.split(",") if o.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========
# Schemas
# =========
class FeatureRecord(BaseModel):
    age: float = Field(..., description="Normalized age")
    sex: float = Field(..., description="Encoded sex feature")
    bmi: float = Field(..., description="Body mass index")
    bp: float = Field(..., description="Mean arterial blood pressure")
    s1: float = Field(..., description="TC")
    s2: float = Field(..., description="LDL")
    s3: float = Field(..., description="HDL")
    s4: float = Field(..., description="TCH")
    s5: float = Field(..., description="LTG")
    s6: float = Field(..., description="GLU")


class PredictRequest(BaseModel):
    records: List[FeatureRecord]
    include_flag: bool = Field(False, description="Return high-risk flags")
    threshold: float | None = Field(None, description="Override threshold (optional)")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "records": [
                    {
                        "age": 0.05,
                        "sex": -0.04,
                        "bmi": 0.03,
                        "bp": 0.02,
                        "s1": 0.01,
                        "s2": 0.0,
                        "s3": -0.01,
                        "s4": 0.02,
                        "s5": 0.03,
                        "s6": -0.02,
                    }
                ],
                "include_flag": True,
            }
        }
    )


class PredictVectorRequest(BaseModel):
    vectors: List[List[float]]  # 10 floats each, in feature_order
    include_flag: bool = Field(False, description="Return high-risk flags")
    threshold: float | None = Field(None, description="Override threshold (optional)")
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Each inner list must have 10 floats in the order shown in /metadata.feature_order",
            "example": {
                "vectors": [
                    [0.05, -0.04, 0.03, 0.02, 0.01, 0.0, -0.01, 0.02, 0.03, -0.02]
                ],
                "include_flag": True,
            },
        }
    )


class PredictResponse(BaseModel):
    scores: List[float]
    feature_order: List[str]
    flags: List[int] | None = None
    threshold_used: float | None = None


class SingleRecord(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


# Example response shown in Swagger (uses loaded defaults)
EXAMPLE_RESPONSE = {
    "scores": [201.82],
    "feature_order": feature_order,
    "flags": [0] if DEFAULT_THR is not None else None,
    "threshold_used": DEFAULT_THR,
}


# ===========
# Root & info
# ===========
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/home", response_class=HTMLResponse, include_in_schema=False)
def home():
    return """
    <h2>Virtual Diabetes Clinic Risk API</h2>
    <ul>
      <li><a href="/docs">Swagger UI</a></li>
      <li><a href="/redoc">ReDoc</a></li>
      <li><a href="/health">Health</a></li>
      <li><a href="/metadata">Metadata</a></li>
    </ul>
    """


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "model": meta, "model_version": meta.get("version")}


@app.get("/live", tags=["ops"])
def live():
    return {"status": "live"}


@app.get("/ready", tags=["ops"])
def ready():
    try:
        _ = pipe.predict(np.zeros((1, len(feature_order))))
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(
            status_code=503, detail={"error": "not_ready", "message": str(e)}
        )


@app.get("/metadata", tags=["model"])
def metadata():
    return {
        "model": meta.get("model"),
        "version": meta.get("version"),
        "feature_order": feature_order,
        "high_risk_threshold": DEFAULT_THR,
        "high_risk_quantile": meta.get("high_risk_quantile"),
        "candidates": meta.get("candidates"),  # includes baseline & alternatives
    }


# ===========
# Inference
# ===========
# ---------- /predict_one ----------
@app.post(
    "/predict_one",
    tags=["inference"],
    summary="Score a single record",
    description="Send one patient record and receive a single score.",
    responses={
        200: {
            "description": "OK",
            "content": {"application/json": {"example": {"prediction": 201.82}}},
        },
        422: {"description": "Validation Error"},
    },
)
def predict_one(rec: SingleRecord):
    try:
        row = [
            [
                rec.age,
                rec.sex,
                rec.bmi,
                rec.bp,
                rec.s1,
                rec.s2,
                rec.s3,
                rec.s4,
                rec.s5,
                rec.s6,
            ]
        ]
        df = pd.DataFrame(row, columns=feature_order)
        score = float(pipe.predict(df)[0])
        return {"prediction": score}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail={"error": "prediction_failed", "message": str(e)}
        )


# ---------- /predict (records as dicts) ----------
@app.post(
    "/predict",
    tags=["inference"],
    response_model=PredictResponse,
    summary="Score records (dict form)",
    description="Send patient records as objects with 10 features. Returns continuous scores and optional high-risk flags.",
    responses={
        200: {
            "description": "OK",
            "content": {"application/json": {"example": EXAMPLE_RESPONSE}},
        },
        400: {"description": "Bad Request (e.g., threshold missing for flags)"},
        422: {"description": "Validation Error"},
    },
)
def predict(
    req: PredictRequest = Body(
        ...,
        openapi_examples={
            "single_record": {
                "summary": "One patient with high-risk flag",
                "value": {
                    "records": [
                        {
                            "age": 0.05,
                            "sex": -0.04,
                            "bmi": 0.03,
                            "bp": 0.02,
                            "s1": 0.01,
                            "s2": 0.0,
                            "s3": -0.01,
                            "s4": 0.02,
                            "s5": 0.03,
                            "s6": -0.02,
                        }
                    ],
                    "include_flag": True,
                },
            }
        },
    )
):
    try:
        rows = [[getattr(rec, col) for col in feature_order] for rec in req.records]
        df = pd.DataFrame(rows, columns=feature_order)
        preds = pipe.predict(df)
        payload: dict = {"scores": preds.tolist(), "feature_order": feature_order}

        if req.include_flag:
            thr = req.threshold if req.threshold is not None else DEFAULT_THR
            if thr is None:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "no_default_threshold",
                        "message": "Model artifact has no high_risk_threshold; provide 'threshold' in request.",
                    },
                )
            flags = (preds >= thr).astype(int).tolist()
            payload.update({"flags": flags, "threshold_used": float(thr)})

        return payload

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail={"error": "prediction_failed", "message": str(e)}
        )


# ---------- /predict_vectors (vectors as lists) ----------
@app.post(
    "/predict_vectors",
    tags=["inference"],
    response_model=PredictResponse,
    summary="Score vectors (list form)",
    description="Send patients as arrays of 10 floats in the exact order of /metadata.feature_order.",
    responses={
        200: {
            "description": "OK",
            "content": {"application/json": {"example": EXAMPLE_RESPONSE}},
        },
        400: {"description": "Bad Request (wrong vector length or threshold missing)"},
        422: {"description": "Validation Error"},
    },
)
def predict_vectors(
    req: PredictVectorRequest = Body(
        ...,
        openapi_examples={
            "one_vector": {
                "summary": "One vector with high-risk flag",
                "value": {
                    "vectors": [
                        [0.05, -0.04, 0.03, 0.02, 0.01, 0.0, -0.01, 0.02, 0.03, -0.02]
                    ],
                    "include_flag": True,
                },
            }
        },
    )
):
    try:
        n = len(feature_order)
        for i, v in enumerate(req.vectors):
            if len(v) != n:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "bad_vector_length",
                        "index": i,
                        "got": len(v),
                        "expected": n,
                    },
                )

        df = pd.DataFrame(req.vectors, columns=feature_order)
        preds = pipe.predict(df)
        payload: dict = {"scores": preds.tolist(), "feature_order": feature_order}

        if req.include_flag:
            thr = req.threshold if req.threshold is not None else DEFAULT_THR
            if thr is None:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "no_default_threshold",
                        "message": "Model artifact has no high_risk_threshold; provide 'threshold' in request.",
                    },
                )
            flags = (preds >= thr).astype(int).tolist()
            payload.update({"flags": flags, "threshold_used": float(thr)})

        return payload

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail={"error": "prediction_failed", "message": str(e)}
        )
