# aero_ai_platform/data_generation/shm.py

import numpy as np
import pandas as pd
from dataclasses import dataclass

from aero_ai_platform.config import SHMConfig


@dataclass
class VibrationSignalModel:
    sample_rate_hz: float = 1000.0  # 1 kHz sampling
    base_freq_hz: float = 50.0      # dominant vibration frequency

    def generate_signal(
        self,
        n_samples: int,
        rng: np.random.Generator,
        damaged: bool = False,
    ) -> np.ndarray:
        t = np.arange(n_samples) / self.sample_rate_hz

        # Base healthy signal: single tone + broadband noise
        signal = (
            1.0 * np.sin(2 * np.pi * self.base_freq_hz * t) +
            0.3 * rng.normal(0, 1, size=n_samples)
        )

        if damaged:
            # Add additional high-frequency components & amplitude modulation
            hf1 = 0.5 * np.sin(2 * np.pi * 200 * t)
            hf2 = 0.3 * np.sin(2 * np.pi * 350 * t)
            envelope = 1.0 + 0.5 * np.sin(2 * np.pi * 2 * t)  # slow amplitude modulation
            signal = envelope * (signal + hf1 + hf2)

        return signal


def generate_shm_dataset(cfg: SHMConfig = SHMConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(seed=999)
    model = VibrationSignalModel()
    rows = []

    for sensor_id in range(cfg.N_SENSORS):
        is_damaged = rng.random() < cfg.DAMAGED_RATIO
        sig = model.generate_signal(cfg.N_SAMPLES_PER_SENSOR, rng, damaged=is_damaged)
        label = "damaged" if is_damaged else "healthy"

        for i in range(cfg.N_SAMPLES_PER_SENSOR):
            rows.append(
                {
                    "sensor_id": sensor_id,
                    "time_idx": i,
                    "accel_g": float(sig[i]),  # pseudo acceleration in g
                    "damage_label": label,
                }
            )

    df = pd.DataFrame(rows)
    return df


def save_shm_dataset(path=None):
    if path is None:
        path = SHMConfig.OUTPUT_PATH
    df = generate_shm_dataset()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[SHM] Saved synthetic dataset to {path}")
