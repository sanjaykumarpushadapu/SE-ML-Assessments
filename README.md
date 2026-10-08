# SE-ML-Assessments

Software engineering for ML assessments.

## Assignment 1: Credit card fraud detection

Folder: [`Assigenment-1/`](Assigenment-1/)

A six-step training pipeline (ingest, clean, features, split, train, evaluate) on the
Kaggle credit card fraud dataset, MLflow tracking and model registry, and a FastAPI
service that serves the registered `fraud-model@champion`. The full write-up is
[`13.md`](Assigenment-1/13.md) / [`13.pdf`](Assigenment-1/13.pdf), and
[`13.ipynb`](Assigenment-1/13.ipynb) runs everything top to bottom.

| Path | What it is |
| --- | --- |
| `config.py` | All paths, the seed and the targets. Every other file reads its settings from here. |
| `pipeline/` | The six filters and `run_pipeline.py`, which chains them and registers the winner in MLflow. |
| `api/` | FastAPI service (`/health`, `/predict`) and the CSV prediction logger. |
| `tests/` | Unit tests. They use a small synthetic table, so they do not need the dataset or MLflow. |
| `artifacts/` | `metrics.json` and `sample_requests.json`, written by the pipeline and read by the report. |
| `docs/screenshots/` | Screenshots used in the report. |
| `build_report.py` | Refreshes the numbers in `13.md`, retakes the screenshots and rebuilds `13.pdf`. |
| `data/creditcard.zip` | The dataset, zipped. |

### Setup

Python 3.12 or newer (the report was produced with 3.14). All commands below run from
inside `Assigenment-1/`.

```bash
cd Assigenment-1
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Unzip the dataset first.** The data ships as `data/creditcard.zip`, but `config.py`
reads `data/creditcard.csv`, and the pipeline stops with `Dataset not found` until the
CSV is there:

```bash
unzip data/creditcard.zip -d data  # creates data/creditcard.csv (about 150 MB, git-ignored)
```

### Run the tests

```bash
pytest -q tests/
```

### Train and register the model

```bash
python -m pipeline.run_pipeline
```

This trains logistic regression and random forest, picks the one with the lowest
validation false-positive rate, evaluates it once on the test set, registers it as
`fraud-model` with the alias `champion`, and writes `artifacts/metrics.json` and
`artifacts/sample_requests.json`. The MLflow data goes to `mlflow.db` and `mlruns/`.
Neither is committed; they are recreated by this command. To start from a clean registry
(as the notebook does), delete them first:

```bash
rm -rf mlruns mlflow.db predictions_log.csv
```

To browse the runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Run the API

The API loads `models:/fraud-model@champion`, so run the pipeline at least once first.

```bash
uvicorn api.main:app --port 8000
```

- `GET  http://127.0.0.1:8000/health` shows the model version and threshold being served.
- `POST http://127.0.0.1:8000/predict` takes the 30 inputs (`Time`, `V1`..`V28`, `Amount`) as JSON.
- Interactive docs are at `http://127.0.0.1:8000/docs`.

Try it with the sample fraud row the pipeline saved:

```bash
python -c "import json; print(json.dumps(json.load(open('artifacts/sample_requests.json'))['fraud']))" \
  | curl -s -X POST http://127.0.0.1:8000/predict -H 'content-type: application/json' -d @-
```

Every prediction is appended to `predictions_log.csv` (git-ignored).

### Run the notebook

```bash
jupyter notebook 13.ipynb
```

Use Restart and Run All. The notebook rewrites the source files with `%%writefile`, runs
the pipeline from scratch, starts the API in the background (its output goes to
`api.log`) and runs the tests.

### Build the report

Run this after a fresh notebook run (Restart and Run All, then save). It refreshes the
numbers in `13.md`, retakes the screenshots in `docs/screenshots/` and rebuilds `13.pdf`.

```bash
pip install markdown playwright    # same as: pip install -r requirements-report.txt
playwright install chromium        # one time: downloads the headless browser
python build_report.py
```

- `python build_report.py --no-shots` keeps the existing screenshots.
- `python build_report.py --no-pdf` only updates `13.md`.

Retaking screenshots starts an MLflow server and the API itself, so `mlflow.db` and
`mlruns/` from a pipeline run must exist, and the machine needs internet access because
the FastAPI docs page loads Swagger UI from a CDN.
