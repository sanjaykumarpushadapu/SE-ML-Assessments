# Credit Card Fraud Detection

AIMLZG546 - Software Engineering for Machine Learning, Assignment I, Group 13.

This project trains Logistic Regression and Random Forest candidates, selects a champion using validation results, registers it in MLflow, and serves fraud predictions through FastAPI. The training pipeline follows the Pipe-and-Filter pattern; the prediction API and model registry run as separate services.

## Requirements

- Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place the extracted CSV at [data/creditcard.csv](data/creditcard.csv). Training needs the real dataset.
- For Docker: install Docker with the Compose plugin and keep the Docker engine running. Local Python is not required for this workflow.
- For the notebook or native workflow: Python 3.14 is the tested version. Use a virtual environment and the project dependencies.

Run all commands from the project root. This is an assignment prototype, not a production payment service.

## Run With Docker

### First Start or Retraining

```text
docker compose up -d --build --wait registry
docker compose run --rm prediction-api python -m pipeline.run_pipeline
docker compose up -d --no-deps --force-recreate --wait prediction-api
docker compose ps
```

The registry must start before training. Training runs inside Docker, selects the champion, and saves the metrics and sample transactions. Recreating the API after training ensures it loads the new champion instead of continuing to serve a previously loaded model.

Both services stay running after these commands and after the final Docker cell in [13.ipynb](13.ipynb) finishes. The `unless-stopped` restart policy lets Docker restart them when its engine restarts, unless they were explicitly stopped.

### Find the Browser URLs

```text
docker compose port prediction-api 8000
docker compose port registry 5000
```

Each command prints a host address such as `127.0.0.1:49152`. Open:

- API documentation: `http://<prediction-api-address>/docs`
- API health: `http://<prediction-api-address>/health`
- MLflow UI: `http://<registry-address>`

Docker assigns available localhost ports to avoid occupied or Windows-reserved ports. Host ports can change when containers are recreated; use the commands above instead of assuming ports 8000 and 5000. Container-to-container communication always uses `http://registry:5000`.

### Restart Without Retraining

Once a champion has been registered and its files are still present:

```text
docker compose up -d --wait
```

### Logs and Shutdown

```text
docker compose logs --tail 100 registry prediction-api
docker compose down
```

Stopping and removing the containers does not delete the database, models, or logs stored in the mounted project folder.

## Database and Storage

The registry runs an embedded SQLite database at `/app/docker-mlflow.db`. SQLite runs inside the registry container; there is no separate PostgreSQL or MySQL container. The project folder is bind-mounted at `/app` in both containers.

| Data | Host location | Purpose |
| --- | --- | --- |
| Docker registry database | `docker-mlflow.db` in the project root | MLflow runs, model versions, and champion alias |
| Native registry database | `mlflow.db` in the project root | Registry used by the native Python workflow |
| Docker model artifacts | `mlruns/` | Saved models for the Docker registry |
| Native model artifacts | `mlruns-native/` | Saved models for the native registry |
| Evaluation and examples | `artifacts/` | Metrics and sample API requests |
| Prediction log | [predictions_log.csv](predictions_log.csv) | One row per successful prediction |

Keep the registry database and corresponding model artifacts together. A database alone cannot restore missing model files. Native and Docker registries use separate databases and model directories. Native notebook training retains `mlflow.db` and `mlruns-native/`, registering a new model version rather than moving or resetting open files. Docker's `mlruns/` and the shared prediction log are preserved. Stop the native API before retraining, then restart it to load the new champion. Evaluation outputs in `artifacts/` are still shared: run training workflows sequentially and regenerate report evidence for the selected registry.

## Notebook Workflow

Open [13.ipynb](13.ipynb) in VS Code or Jupyter and run the cells in order. It covers data exploration, generated source files, training, MLflow registration, API checks, tests, and optional Docker deployment.

The notebook's `%%writefile` cells generate the Python modules and Docker configuration. When changing a generated file, also update its notebook cell so rerunning the notebook does not overwrite the change.

MLflow records copied from another machine may contain artifact paths that do not exist locally. If this happens, stop native services and restart the kernel before manually archiving the imported native database and artifacts; then retrain with local paths. Normal notebook training does not archive or move them automatically. Docker is optional for the native workflow.

## Native Python Workflow

### Windows PowerShell

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pipeline.run_pipeline
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### macOS or Linux

Using a Python 3.14 interpreter:

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pipeline.run_pipeline
.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`. If port 8000 is unavailable, choose another available port with `--port` and use that port in the URL. The notebook's native API cell selects an available port automatically. Stop the terminal-launched API with Ctrl+C.

Train before starting the API. Never move an open native database: Windows can reject this with `WinError 32`. The native API uses the local registry by default; its tracking URI can be configured through `MLFLOW_TRACKING_URI` in [config.py](config.py).

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Reports service status, model version, and threshold |
| `POST /predict` | Predicts fraud for one transaction |
| `GET /docs` | Interactive API documentation |

A prediction request contains `Time`, `V1` through `V28`, and `Amount`. Use the `fraud` or `genuine` object from [artifacts/sample_requests.json](artifacts/sample_requests.json) as the request body in the interactive docs. Do not send the entire sample file or the target column `Class`.

The response includes the request ID, fraud probability, fraud flag, threshold, model version, and latency. Invalid inputs return HTTP 422. Successful predictions are written to the prediction log.

At startup, the API resolves the champion alias once and loads that immutable model version and its threshold. A subsequent promotion takes effect only after the API is restarted.

The current test recall is 85.26%, below the 90% acceptance target. The demonstrator is not accepted for production payment decisions. Future model and threshold changes must use training/validation data, followed by a frozen evaluation on fresh, untouched hold-out data; do not tune against the reported test results.

## Tests

Windows:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:warnings tests/
```

macOS or Linux:

```sh
.venv/bin/python -m pytest -q -p no:warnings tests/
```

## Manual Report Generation

Run and save [13.ipynb](13.ipynb), then run [build_report.py](build_report.py) separately from the terminal to regenerate [13.md](13.md), screenshots, and [13.pdf](13.pdf). This reporting utility is not part of the assignment notebook's executable code. Report generation needs local Python even when training and serving use Docker.

For a terminal refresh on Windows, install the report tools once and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-report.txt
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe build_report.py
```

On macOS or Linux, use `.venv/bin/python` instead. The default command refreshes all generated report sections, screenshots, and the PDF together; `--no-shots` and `--no-pdf` intentionally skip parts of that refresh.

The report's deployment instructions come from the saved notebook's Docker section. Playwright discovers the current Docker ports and captures the live API and registry, Compose status, and read-only SQLite evidence. If saved notebook versions are older than the metrics, the builder can verify the current metrics against the live Docker API and champion's MLflow run instead. The model version, threshold, selected model, and logged evaluation metrics must agree; otherwise generation stops with a diagnostic. Older notebook outputs remain labelled as historical evidence. Services remain running. Screenshot 1 shows only the six pipeline stages and a compact evaluation table, without Docker build logs. The PDF omits the full-code appendix and is exported only after images and Mermaid diagrams render. Native prediction examples and latency measurements are labelled separately from Docker registry evidence.

Save the notebook and repeat the refresh whenever notebook code or results change. The builder updates only its marked generated sections; manually written report text remains editable.

## Project Files

- [13.ipynb](13.ipynb): executable notebook and source-generation cells.
- [13.md](13.md): assignment report, design decisions, and evaluation results.
- [pipeline/run_pipeline.py](pipeline/run_pipeline.py): chains ingest, clean, features, split, train, and evaluate.
- [api/main.py](api/main.py): prediction API and champion loading.
- [config.py](config.py): shared paths, model settings, and registry configuration.
- [docker-compose.yml](docker-compose.yml): API and registry services, health checks, and persistence.
- [Dockerfile](Dockerfile): shared Python container image.
- [requirements.txt](requirements.txt): native training, API, notebook, and test dependencies.
- [requirements-api.txt](requirements-api.txt): container dependencies.
- [build_report.py](build_report.py): report generation; additional dependencies are in [requirements-report.txt](requirements-report.txt).

## Troubleshooting

- **API cannot load the champion:** run the Docker training command before starting the API, or retrain the native registry through the notebook after moving machines. Preserve both the registry database and model artifacts.
- **Docker services are not running:** ensure the Docker engine is running, inspect `docker compose ps` and the service logs, then use the startup commands above.
- **Cannot open the API URL:** discover the current host port with `docker compose port prediction-api 8000`. The services bind to localhost and are not exposed to other devices on the network.
- **Missing dataset:** verify the extracted CSV is at the expected dataset path before training.