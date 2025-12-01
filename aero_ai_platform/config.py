# aero_ai_platform/config.py

from pathlib import Path


# Base project path (assumes this file is inside aero_ai_platform/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Model artifacts directory
MODELS_DIR = BASE_DIR / "models_artifacts"

# Create directories at import time (simple MVP approach)
for p in [DATA_DIR, RAW_DATA_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


class FlightConfig:
    N_FLIGHTS = 500  # number of synthetic flights
    N_POINTS_PER_FLIGHT = 300  # time steps per flight
    ANOMALY_RATIO = 0.1  # 10% flights contain anomalies
    OUTPUT_PATH = RAW_DATA_DIR / "flight_data.csv"


class EngineConfig:
    N_ENGINES = 50
    N_CYCLES_PER_ENGINE = 200
    FAULT_RATIO = 0.2
    OUTPUT_PATH = RAW_DATA_DIR / "engine_data.csv"


class SHMConfig:
    N_SENSORS = 20
    N_SAMPLES_PER_SENSOR = 5000  # time steps of vibration signal
    DAMAGED_RATIO = 0.2
    OUTPUT_PATH = RAW_DATA_DIR / "shm_data.csv"
