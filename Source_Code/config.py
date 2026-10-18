from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_PATH = DATA_DIR / "thyroid_data.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20
for folder in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
    folder.mkdir(parents=True, exist_ok=True)
