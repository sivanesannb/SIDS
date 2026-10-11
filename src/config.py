"""Central settings for the Smart Irrigation project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "train.csv"      # Kaggle Playground S6E4 train.csv
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

TARGET = "irrigation_need"
CLASSES = ["Low", "Medium", "High"]           # ordinal order
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
RANDOM_STATE = 42

# ---- Water-saving simulation assumptions (state these in your report) ----
# Assumed irrigation depth (mm) applied per event for each predicted need level.
DOSE_MM = {"Low": 10, "Medium": 30, "High": 60}
# Baseline = fixed schedule: farmer applies the same "Medium" dose to every field.
BASELINE_DOSE_MM = 30
