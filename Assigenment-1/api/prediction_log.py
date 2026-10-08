"""Non-ML component: append one row per prediction to a CSV file (BR-008)."""
import csv
import threading
from datetime import datetime, timezone
from pathlib import Path

import config

FIELDS = ["timestamp", "request_id", "probability", "is_fraud", "latency_ms", "model_version"]
_lock = threading.Lock()  # two requests at the same time must not write into the file together


def log_prediction(request_id: str, probability: float, is_fraud: bool, latency_ms: float,
                   model_version: str, path: Path | None = None) -> None:
    """Write one log row; an OSError is raised if the file cannot be written."""
    path = Path(path or config.LOG_PATH)
    # Only result data is logged. The transaction values themselves are not stored.
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), "request_id": request_id,
           "probability": round(probability, 6), "is_fraud": int(is_fraud),
           "latency_ms": round(latency_ms, 3), "model_version": model_version}
    with _lock:
        new_file = not path.exists() or path.stat().st_size == 0  # write the header once
        with path.open("a", newline="") as f:  # "a" = append, old rows are kept
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
