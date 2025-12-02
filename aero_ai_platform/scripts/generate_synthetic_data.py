<<<<<<< HEAD
# scripts/generate_synthetic_data.py

#!/usr/bin/env python
"""
Generate all synthetic datasets (flight, engine, shm) in one go.
Run: python scripts/generate_synthetic_data.py
"""

import sys
from pathlib import Path

# Add parent directory to path so imports work when running directly
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from aero_ai_platform.data_generation.flight import save_flight_dataset
from aero_ai_platform.data_generation.engine import save_engine_dataset
from aero_ai_platform.data_generation.shm import save_shm_dataset


def main():
    print("Generating synthetic datasets for Aero AI Platform (offline MVP)...")
    save_flight_dataset()
    save_engine_dataset()
    save_shm_dataset()
    print("Done.")


if __name__ == "__main__":
    main()
=======
# scripts/generate_synthetic_data.py

#!/usr/bin/env python
"""
Generate all synthetic datasets (flight, engine, shm) in one go.
Run: python scripts/generate_synthetic_data.py
"""

import sys
from pathlib import Path

# Add parent directory to path so imports work when running directly
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from aero_ai_platform.data_generation.flight import save_flight_dataset
from aero_ai_platform.data_generation.engine import save_engine_dataset
from aero_ai_platform.data_generation.shm import save_shm_dataset


def main():
    print("Generating synthetic datasets for Aero AI Platform (offline MVP)...")
    save_flight_dataset()
    save_engine_dataset()
    save_shm_dataset()
    print("Done.")


if __name__ == "__main__":
    main()
>>>>>>> 5d6d902cff8aafb7cae4854f0b1fef4c6e014b4d
