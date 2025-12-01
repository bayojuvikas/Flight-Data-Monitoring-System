# aero_ai_platform/models/shm.py

import joblib
import pandas as pd
from typing import Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from aero_ai_platform.config import MODELS_DIR
from aero_ai_platform.features.shm_features import build_shm_features


MODEL_PATH = MODELS_DIR / "shm_rf_model.pkl"


def _prepare_shm_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    df = build_shm_features()

    # binary label: 1 damaged, 0 healthy
    y = (df["damage_label"] == "damaged").astype(int)

    drop_cols = ["sensor_id", "damage_label"]
    X = df.drop(columns=drop_cols)

    return X, y


def train_shm_model(save: bool = True):
    X, y = _prepare_shm_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("=== SHM damage model report ===")
    print(classification_report(y_test, y_pred, digits=3))

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "feature_names": list(X.columns),
            },
            MODEL_PATH,
        )
        print(f"[SHM] Saved model to {MODEL_PATH}")


def load_shm_model():
    obj = joblib.load(MODEL_PATH)
    return obj["model"], obj["feature_names"]
