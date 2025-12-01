# aero_ai_platform/features/shm_features.py

import pandas as pd
import numpy as np

from aero_ai_platform.config import SHMConfig


def load_shm_raw(path=None) -> pd.DataFrame:
    if path is None:
        path = SHMConfig.OUTPUT_PATH
    df = pd.read_csv(path)
    return df


def _sensor_features(accel: np.ndarray, sample_rate_hz: float = 1000.0) -> dict:
    """
    Compute basic time-domain and simple frequency-domain features.
    """
    features: dict[str, float] = {}
    if accel.size == 0:
        return {
            "rms": 0.0,
            "std": 0.0,
            "max_abs": 0.0,
            "energy": 0.0,
            "hf_ratio": 0.0,
        }

    features["rms"] = np.sqrt(np.mean(accel**2))
    features["std"] = float(np.std(accel))
    features["max_abs"] = float(np.max(np.abs(accel)))
    features["energy"] = float(np.sum(accel**2))

    # simple frequency feature: ratio of high-frequency energy (>150 Hz)
    n = accel.size
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    spectrum = np.abs(np.fft.rfft(accel))

    total_power = np.sum(spectrum)
    hf_power = np.sum(spectrum[freqs > 150.0])
    features["hf_ratio"] = float(hf_power / (total_power + 1e-8))

    return features


def build_shm_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Aggregate time-series accel_g into per-sensor features.
    Returns one row per sensor_id.
    """
    if df is None:
        df = load_shm_raw()

    df = df.sort_values(["sensor_id", "time_idx"])

    rows = []
    for sensor_id, g in df.groupby("sensor_id"):
        accel = g["accel_g"].to_numpy(dtype=float)
        feats = _sensor_features(accel)
        feats["sensor_id"] = sensor_id
        # label is constant per sensor in our synthetic data
        feats["damage_label"] = g["damage_label"].iloc[0]
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
    return feat_df


def save_shm_features(path=None):
    if path is None:
        path = SHMConfig.OUTPUT_PATH.with_name("shm_features.csv")
    df_feat = build_shm_features()
    path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(path, index=False)
    print(f"[SHM] Saved feature matrix to {path}")
