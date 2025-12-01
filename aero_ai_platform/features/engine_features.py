# aero_ai_platform/features/engine_features.py

import pandas as pd

from aero_ai_platform.config import EngineConfig


def load_engine_raw(path=None) -> pd.DataFrame:
    if path is None:
        path = EngineConfig.OUTPUT_PATH
    df = pd.read_csv(path)
    return df


def build_engine_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build per-engine-cycle features for RUL regression.
    One row per engine_id + cycle.
    """
    if df is None:
        df = load_engine_raw()

    # ensure sorting
    df = df.sort_values(["engine_id", "cycle"])

    # compute deltas within each engine
    df["egt_delta"] = df.groupby("engine_id")["egt_c"].diff().fillna(0.0)
    df["ff_delta"] = df.groupby("engine_id")["fuel_flow_kgph"].diff().fillna(0.0)

    # we keep everything; you can also filter or normalize later
    return df


def save_engine_features(path=None):
    if path is None:
        path = EngineConfig.OUTPUT_PATH.with_name("engine_features.csv")
    df_feat = build_engine_features()
    path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(path, index=False)
    print(f"[Engine] Saved feature matrix to {path}")
