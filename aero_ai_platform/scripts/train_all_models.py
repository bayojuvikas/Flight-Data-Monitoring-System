# scripts/train_all_models.py

#!/usr/bin/env python
"""
Train all three models (flight anomaly, engine RUL, SHM damage)
using the synthetic datasets.

Run:
    python scripts/generate_synthetic_data.py
    python scripts/train_all_models.py
"""

import sys
from pathlib import Path

# Add parent directory to path so imports work when running directly
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from aero_ai_platform.models.flight import train_flight_model
from aero_ai_platform.models.engine import train_engine_model
from aero_ai_platform.models.shm import train_shm_model


def main():
    print("Training Flight anomaly model...")
    train_flight_model()

    print("\nTraining Engine RUL model...")
    train_engine_model()

    print("\nTraining SHM damage model...")
    train_shm_model()

    print("\nAll models trained and saved.")


if __name__ == "__main__":
    main()
