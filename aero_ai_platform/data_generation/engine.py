# aero_ai_platform/data_generation/engine.py

import numpy as np
import pandas as pd
from dataclasses import dataclass

from aero_ai_platform.config import EngineConfig


@dataclass
class EngineDegradationModel:
    base_egt: float = 600.0  # °C
    base_n1: float = 90.0    # %
    base_fuel_flow: float = 2500.0  # kg/h

    def generate_engine_cycles(
        self,
        n_cycles: int,
        rng: np.random.Generator,
        faulty: bool = False,
    ) -> pd.DataFrame:
        cycles = np.arange(n_cycles)

        # Degradation over cycles
        egt_trend = self.base_egt + (cycles * 0.3)  # gradual EGT rise
        n1_trend = self.base_n1 - (cycles * 0.01)   # slight efficiency loss
        ff_trend = self.base_fuel_flow + (cycles * 0.5)  # more fuel to achieve thrust

        # Add noise
        egt = egt_trend + rng.normal(0, 5, size=n_cycles)
        n1 = n1_trend + rng.normal(0, 0.3, size=n_cycles)
        ff = ff_trend + rng.normal(0, 20, size=n_cycles)

        fault_label = "healthy"
        # If faulty, add a steeper change after some cycle
        if faulty:
            fault_label = "faulty"
            fault_start = rng.integers(low=int(n_cycles * 0.4), high=int(n_cycles * 0.7))
            egt[fault_start:] += np.linspace(20, 100, n_cycles - fault_start)
            ff[fault_start:] += np.linspace(50, 200, n_cycles - fault_start)

        # Approximate Remaining Useful Life (RUL)
        egt_threshold = self.base_egt + 80
        egt_margin = np.maximum(egt_threshold - egt, 1)
        rul = (egt_margin / 0.5).clip(min=1)  # synthetic RUL in cycles

        df = pd.DataFrame(
            {
                "cycle": cycles,
                "egt_c": egt,
                "n1_pct": n1,
                "fuel_flow_kgph": ff,
                "fault_label": fault_label,
                "rul_cycles": rul,
            }
        )
        return df


def generate_engine_dataset(cfg: EngineConfig = EngineConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(seed=123)
    rows = []
    model = EngineDegradationModel()

    for engine_id in range(cfg.N_ENGINES):
        is_faulty = rng.random() < cfg.FAULT_RATIO
        df_engine = model.generate_engine_cycles(cfg.N_CYCLES_PER_ENGINE, rng, faulty=is_faulty)
        df_engine["engine_id"] = engine_id
        rows.append(df_engine)

    df = pd.concat(rows, ignore_index=True)
    return df


def save_engine_dataset(path=None):
    if path is None:
        path = EngineConfig.OUTPUT_PATH
    df = generate_engine_dataset()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[Engine] Saved synthetic dataset to {path}")
