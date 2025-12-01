# aero_ai_platform/models/engine.py

import joblib
import pandas as pd
from typing import Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from aero_ai_platform.config import MODELS_DIR
from aero_ai_platform.features.engine_features import build_engine_features


MODEL_PATH = MODELS_DIR / "engine_rul_rf_model.pkl"


def _prepare_engine_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    df = build_engine_features()

    # Targets: RUL in cycles
    y = df["rul_cycles"]

    # Remove non-feature columns
    drop_cols = ["engine_id", "fault_label", "rul_cycles"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]

    return X, y


def train_engine_model(save: bool = True):
    X, y = _prepare_engine_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("=== Engine RUL model report ===")
    print(f"MSE: {mse:.3f}")
    print(f"R^2: {r2:.3f}")

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "feature_names": list(X.columns),
            },
            MODEL_PATH,
        )
        print(f"[Engine] Saved model to {MODEL_PATH}")


def load_engine_model():
    obj = joblib.load(MODEL_PATH)
    return obj["model"], obj["feature_names"]
