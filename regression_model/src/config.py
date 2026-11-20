from pathlib import Path

# resolve data paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "model.csv"
