"""Single place for paths, constants and targets (BR-002, BR-011).

Every other file imports its settings from here, so a value such as the seed or the
target recall is never typed in two different places.
"""
import os
from pathlib import Path

# Folder that contains this file (the project folder, i.e. where the notebook runs).
ROOT = Path(__file__).resolve().parent
SEED = 42  # fixed seed so every run gives the same split and the same model (BR-011)

# Where the data, outputs and the prediction log live.
DATA_PATH = ROOT / "data" / "creditcard.csv"
ARTIFACTS_DIR = ROOT / "artifacts"       # metrics.json and sample requests are saved here
LOG_PATH = ROOT / "predictions_log.csv"  # written by log_prediction() on every API call

# Native MLflow uses local SQLite and mlruns-native; Docker overrides both storage settings.
# In Docker Compose the API reaches the registry service over HTTP instead (MLFLOW_TRACKING_URI, Section 8).
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
ARTIFACT_DIR = Path(os.environ.get("MLFLOW_ARTIFACT_DIR", str(ROOT / "mlruns-native"))).resolve()
ARTIFACT_ROOT = ARTIFACT_DIR.as_uri()
EXPERIMENT = "fraud-detection"
MODEL_NAME = "fraud-model"  # name in the model registry
ALIAS = "champion"          # the API only ever loads the model that has this alias (BR-009)

# Column names of the Kaggle dataset: Time, V1..V28 (anonymised), Amount; label is Class.
V_COLS = [f"V{i}" for i in range(1, 29)]
FEATURES = ["Time"] + V_COLS + ["Amount"]
SCALED_COLS = ["Time", "Amount"]  # only these two are on very different scales, so only they are scaled
TARGET = "Class"                  # 1 = fraud, 0 = genuine

# Measurable goals from the report (Section 2.2).
TARGET_RECALL = 0.90  # catch at least 90% of the frauds
MAX_FPR = 0.02        # wrongly flag less than 2% of genuine transactions
MIN_LOSS_REDUCTION = 0.30  # business goal: cut fraud loss by 30% (measured as the share of fraud money caught)
MAX_P95_MS = 200.0    # 95% of API calls must answer within 200 ms
