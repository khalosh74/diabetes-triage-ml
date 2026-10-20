import json
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
import joblib
import numpy as np

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
    return model, ModelReport(
        name=name, rmse=rmse, precision_at_thr=prec, recall_at_thr=rec
    )


def main():
    # Separate model selection from final evaluation:
    # 60% fit, 20% validation, 20% untouched test.
    dataset = load_diabetes(as_frame=True)
    df = dataset.frame.copy()
    X = df.drop(columns=["target"])
    y = df["target"]

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=RANDOM_STATE
    )
    numeric_cols = X.columns.tolist()

    lin = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]
    )
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    )

    # Derive the risk threshold only from the fit data.
    high_risk_threshold = float(np.quantile(y_fit, HIGH_RISK_Q))

    # Select between Ridge and RF on validation data; never on the test set.
    _, val_lin = _train_and_eval(
        lin, X_fit, y_fit, X_val, y_val, high_risk_threshold, "LinearRegression"
    )
    _, val_ridge = _train_and_eval(
        ridge, X_fit, y_fit, X_val, y_val, high_risk_threshold, "Ridge"
    )
    _, val_rf = _train_and_eval(
        rf, X_fit, y_fit, X_val, y_val, high_risk_threshold, "RandomForestRegressor"
    )
    chosen = min([val_ridge, val_rf], key=lambda report: report.rmse)

    # Refit on all development examples, then evaluate the untouched test once.
    selected_model = ridge if chosen.name == "Ridge" else rf
    selected_model, test_selected = _train_and_eval(
        selected_model,
        X_dev,
        y_dev,
        X_test,
        y_test,
        high_risk_threshold,
        chosen.name,
    )
    _, test_baseline = _train_and_eval(
        lin,
        X_dev,
        y_dev,
        X_test,
        y_test,
        high_risk_threshold,
        "LinearRegression",
    )

    validation_reports = [val_lin, val_ridge, val_rf]
    bundle = {
        "pipeline": selected_model,
        "feature_order": numeric_cols,
        "metadata": {
            "model": chosen.name,
            "version": VERSION,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "validation_fraction_of_development": 0.25,
            "high_risk_quantile": HIGH_RISK_Q,
            "high_risk_threshold": high_risk_threshold,
            "candidates": [asdict(report) for report in validation_reports],
            "test_result": asdict(test_selected),
        },
    }
    joblib.dump(bundle, ARTIFACT_DIR / "model.pkl")

    # Preserve legacy report keys, documenting that individual candidates
    # were measured on validation; baseline and final_test use held-out test.
    metrics = {
        "version": VERSION,
        "selection_split": "validation",
        "test_evaluated_after_selection": True,
        "split_fractions": {"fit": 0.6, "validation": 0.2, "test": 0.2},
        "threshold_value": high_risk_threshold,
        "baseline": asdict(test_baseline),
        "ridge": asdict(val_ridge),
        "random_forest": asdict(val_rf),
        "validation_baseline": asdict(val_lin),
        "final_test": asdict(test_selected),
        "selected": chosen.name,
        "deltas_vs_baseline": {
            "rmse_delta": test_selected.rmse - test_baseline.rmse,
            "precision_delta": (
                (test_selected.precision_at_thr or 0)
                - (test_baseline.precision_at_thr or 0)
            ),
            "recall_delta": (
                (test_selected.recall_at_thr or 0) - (test_baseline.recall_at_thr or 0)
            ),
        },
    }
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    change = (
        textwrap.dedent(
            f"""
        ## v{VERSION}

        - Model selected on validation RMSE: **{chosen.name}**
        - Risk threshold (fit-set {int(HIGH_RISK_Q * 100)}th percentile): **{high_risk_threshold:.3f}**
        - Hold-out RMSE (baseline): {test_baseline.rmse:.3f}
        - Hold-out RMSE (selected): **{test_selected.rmse:.3f}**  _(Δ: {test_selected.rmse - test_baseline.rmse:+.3f})_
        - Hold-out precision @ threshold (baseline → selected): {(test_baseline.precision_at_thr or 0):.3f} → **{(test_selected.precision_at_thr or 0):.3f}**
        - Hold-out recall @ threshold (baseline → selected): {(test_baseline.recall_at_thr or 0):.3f} → **{(test_selected.recall_at_thr or 0):.3f}**
        """
        ).strip()
        + "\n"
    )
    current = CHANGELOG.read_text() if CHANGELOG.exists() else "# Changelog\n\n"
    if change not in current:
        if not current.endswith("\n"):
            current += "\n"
        CHANGELOG.write_text(current + change + "\n")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
