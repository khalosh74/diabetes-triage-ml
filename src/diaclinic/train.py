import json
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, precision_score, recall_score

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

CHANGELOG = Path("CHANGELOG.md")

RANDOM_STATE = 42
TEST_SIZE = 0.2
VERSION = "0.2.0"
HIGH_RISK_Q = 0.75  # high-risk = top quartile of progression index

@dataclass
class ModelReport:
    name: str
    rmse: float
    precision_at_thr: float | None = None
    recall_at_thr: float | None = None

def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _train_and_eval(model, X_tr, y_tr, X_te, y_te, thr_val=None, name="model"):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    rmse = _rmse(y_te, preds)
    prec = rec = None
    if thr_val is not None:
        y_true_flag = (y_te >= thr_val).astype(int)
        y_pred_flag = (preds >= thr_val).astype(int)
        prec = float(precision_score(y_true_flag, y_pred_flag, zero_division=0))
        rec = float(recall_score(y_true_flag, y_pred_flag, zero_division=0))
    return model, ModelReport(name=name, rmse=rmse, precision_at_thr=prec, recall_at_thr=rec)

def main():
    # ===== 1) load and split =====
    Xy = load_diabetes(as_frame=True)
    df = Xy.frame.copy()
    X = df.drop(columns=["target"])
    y = df["target"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    numeric_cols = X.columns.tolist()

    # ===== 2) define candidates =====
    lin = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE))])
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    )  # RF doesn’t need scaling

    # High-risk threshold: top quartile of *train* y
    high_risk_threshold = float(np.quantile(y_tr, HIGH_RISK_Q))

    # ===== 3) train + evaluate all =====
    _, rep_lin = _train_and_eval(lin, X_tr, y_tr, X_te, y_te, high_risk_threshold, "LinearRegression")
    _, rep_ridge = _train_and_eval(ridge, X_tr, y_tr, X_te, y_te, high_risk_threshold, "Ridge")
    best_model, rep_rf = _train_and_eval(rf, X_tr, y_tr, X_te, y_te, high_risk_threshold, "RandomForestRegressor")

    # Pick the *lowest RMSE* among Ridge / RF (Iteration 2 improvement)
    candidates = [rep_ridge, rep_rf]
    best = min(candidates, key=lambda r: r.rmse)
    best_name = best.name
    if best_name == "Ridge":
        best_model = ridge.fit(X_tr, y_tr)  # ensure fitted instance
    else:
        best_model = rf.fit(X_tr, y_tr)

    # ===== 4) persist bundle =====
    bundle = {
        "pipeline": best_model,              # may be Pipeline or RF
        "feature_order": numeric_cols,
        "metadata": {
            "model": best_name,
            "version": VERSION,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "high_risk_quantile": HIGH_RISK_Q,
            "high_risk_threshold": high_risk_threshold,
            "candidates": [asdict(rep_lin), asdict(rep_ridge), asdict(rep_rf)],
        },
    }
    joblib.dump(bundle, ARTIFACT_DIR / "model.pkl")

    # ===== 5) save metrics.json (best + baseline + deltas) =====
    metrics = {
        "version": VERSION,
        "threshold_value": high_risk_threshold,
        "baseline": asdict(rep_lin),
        "ridge": asdict(rep_ridge),
        "random_forest": asdict(rep_rf),
        "selected": best_name,
        "deltas_vs_baseline": {
            "rmse_delta": (best.rmse - rep_lin.rmse),               # negative = improvement
            "precision_delta": (
                (best.precision_at_thr or 0) - (rep_lin.precision_at_thr or 0)
            ),
            "recall_delta": (
                (best.recall_at_thr or 0) - (rep_lin.recall_at_thr or 0)
            ),
        },
    }
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ===== 6) append CHANGELOG =====
    change = textwrap.dedent(
        f"""
        ## v{VERSION}

        - Model: **{best_name}**
        - High-risk threshold (train {int(HIGH_RISK_Q*100)}th pct of y): **{high_risk_threshold:.3f}**
        - RMSE (baseline LinearRegression): {rep_lin.rmse:.3f}
        - RMSE (selected): **{best.rmse:.3f}**  _(Δ vs baseline: {best.rmse - rep_lin.rmse:+.3f}, lower is better)_
        - Precision @ threshold (baseline → selected): {(rep_lin.precision_at_thr or 0):.3f} → **{(best.precision_at_thr or 0):.3f}**
        - Recall @ threshold (baseline → selected): {(rep_lin.recall_at_thr or 0):.3f} → **{(best.recall_at_thr or 0):.3f}**
        """
    ).strip() + "\n"

    if CHANGELOG.exists():
        current = CHANGELOG.read_text()
    else:
        current = "# Changelog\n\n"

    if change not in current:
        if not current.endswith("\n"):
            current += "\n"
        CHANGELOG.write_text(current + change + "\n")

    # console hint
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
