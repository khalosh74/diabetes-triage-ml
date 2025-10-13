import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

def main():
    Xy = load_diabetes(as_frame=True)
    df = Xy.frame.copy()
    X = df.drop(columns=["target"]) #traing set
    y = df["target"] #Target set

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    numeric_cols = X.columns.tolist()
    print(numeric_cols)
    
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    joblib.dump(
        {
            "pipeline": pipe,
            "feature_order": numeric_cols,
            "metadata": {
                "model": "LinearRegression",
                "version": "0.1.0",
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
            },
        },
        ARTIFACT_DIR / "model.pkl",
    )

    metrics = {"rmse": rmse}
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Saved model to {ARTIFACT_DIR/'model.pkl'} (RMSE={rmse:.3f})")

if __name__ == "__main__":
    main()
