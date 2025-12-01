# aero_ai_platform/features/flight_features.py

import pandas as pd

from aero_ai_platform.config import FlightConfig


def load_flight_raw(path=None) -> pd.DataFrame:
    if path is None:
        path = FlightConfig.OUTPUT_PATH
    df = pd.read_csv(path)
    return df


def build_flight_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Aggregate time-series flight data into per-flight feature vectors.
    Returns one row per flight_id with summary statistics.
    """
    if df is None:
        df = load_flight_raw()

    # Sort for consistent slicing
    df = df.sort_values(["flight_id", "time_idx"])

    feature_rows = []

    for flight_id, g in df.groupby("flight_id"):
        n = len(g)
        # last 20% of the flight = "approach" proxy
        start_approach = int(n * 0.8)
        g_approach = g.iloc[start_approach:]

        def stats(prefix: str, series: pd.Series):
            return {
                f"{prefix}_mean": series.mean(),
                f"{prefix}_std": series.std(),
                f"{prefix}_min": series.min(),
                f"{prefix}_max": series.max(),
            }

        features = {
            "flight_id": flight_id,
        }

        # Global stats
        features.update(stats("alt", g["altitude_m"]))
        features.update(stats("ias", g["ias_kt"]))
        features.update(stats("pitch", g["pitch_deg"]))
        features.update(stats("roll", g["roll_deg"]))
        features.update(stats("n1", g["engine_n1_pct"]))

        # Approach variability (key for unstable approaches)
        features["approach_alt_std"] = g_approach["altitude_m"].std()
        features["approach_ias_std"] = g_approach["ias_kt"].std()

        # Label (same for whole flight)
        label = g["anomaly_label"].iloc[0]
        features["anomaly_label"] = label

        feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows)
    return features_df


def save_flight_features(path=None):
    if path is None:
        path = FlightConfig.OUTPUT_PATH.with_name("flight_features.csv")
    df_feat = build_flight_features()
    path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(path, index=False)
    print(f"[Flight] Saved feature matrix to {path}")
