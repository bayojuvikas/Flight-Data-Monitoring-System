# aero_ai_platform/data_generation/flight.py

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

from aero_ai_platform.config import FlightConfig


@dataclass
class FlightPhaseProfile:
    """Simple piecewise profile for altitude & airspeed."""
    n_points: int

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            altitude (meters), ias (knots)
        """
        n = self.n_points

        # segments: climb (0-30%), cruise (30-70%), descent (70-100%)
        t = np.linspace(0, 1, n)

        # Altitude profile (in meters)
        alt = np.piecewise(
            t,
            [t < 0.3, (t >= 0.3) & (t < 0.7), t >= 0.7],
            [
                lambda x: 0 + 12000 * (x / 0.3),        # climb to ~12,000 m
                lambda x: 12000 + 200 * np.sin(5 * x),  # slight cruise oscillation
                lambda x: 12000 * (1 - (x - 0.7) / 0.3) # descent to ~0
            ],
        )

        # IAS profile (in knots)
        ias = np.piecewise(
            t,
            [t < 0.3, (t >= 0.3) & (t < 0.7), t >= 0.7],
            [
                lambda x: 160 + 60 * (x / 0.3),     # climb accelerating to ~220
                lambda x: 220 + 10 * np.sin(6 * x), # cruise around 220
                lambda x: 220 - 80 * ((x - 0.7) / 0.3),  # slow to ~140 on approach
            ],
        )

        return alt, ias


def inject_flight_anomaly(
    alt: np.ndarray,
    ias: np.ndarray,
    anomaly_type: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Inject anomalies:
    - 'unstable_approach': altitude and ias oscillate near end
    - 'overspeed_climb': excessive IAS during climb
    """
    n = len(alt)
    alt = alt.copy()
    ias = ias.copy()
    label = anomaly_type

    if anomaly_type == "unstable_approach":
        start = int(n * 0.8)
        noise_alt = rng.normal(0, 400, size=n - start)  # ±400 m oscillations
        noise_ias = rng.normal(0, 20, size=n - start)   # ±20 kt oscillations
        alt[start:] += noise_alt
        ias[start:] += noise_ias

    elif anomaly_type == "overspeed_climb":
        end = int(n * 0.3)
        ias[:end] += rng.normal(50, 10, size=end)  # +50 kt in climb

    else:
        label = "normal"

    return alt, ias, label


def generate_flight_dataset(cfg: FlightConfig = FlightConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    rows = []

    for flight_id in range(cfg.N_FLIGHTS):
        phase = FlightPhaseProfile(cfg.N_POINTS_PER_FLIGHT)
        base_alt, base_ias = phase.generate()

        # Add base noise
        alt = base_alt + rng.normal(0, 50, size=cfg.N_POINTS_PER_FLIGHT)  # ±50m noise
        ias = base_ias + rng.normal(0, 5, size=cfg.N_POINTS_PER_FLIGHT)   # ±5kt noise

        # Decide anomaly
        is_anomaly = rng.random() < cfg.ANOMALY_RATIO
        anomaly_type = "normal"
        if is_anomaly:
            anomaly_type = rng.choice(["unstable_approach", "overspeed_climb"])
            alt, ias, _ = inject_flight_anomaly(alt, ias, anomaly_type, rng)

        # Generate some simple attitude + engine proxy signals
        t = np.arange(cfg.N_POINTS_PER_FLIGHT)
        pitch = np.clip(10 * np.sin(t / 40.0) + rng.normal(0, 1, size=t.size), -10, 15)
        roll = np.clip(5 * np.sin(t / 30.0) + rng.normal(0, 1, size=t.size), -30, 30)
        engine_n1 = 85 + 5 * np.sin(t / 50.0) + rng.normal(0, 1, size=t.size)  # %

        for i in range(cfg.N_POINTS_PER_FLIGHT):
            rows.append(
                {
                    "flight_id": flight_id,
                    "time_idx": i,
                    "altitude_m": float(alt[i]),
                    "ias_kt": float(ias[i]),
                    "pitch_deg": float(pitch[i]),
                    "roll_deg": float(roll[i]),
                    "engine_n1_pct": float(engine_n1[i]),
                    "anomaly_label": anomaly_type,
                }
            )

    df = pd.DataFrame(rows)
    return df


def save_flight_dataset(path=None):
    if path is None:
        path = FlightConfig.OUTPUT_PATH
    df = generate_flight_dataset()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[Flight] Saved synthetic dataset to {path}")
