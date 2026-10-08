# One image for both services (registry and prediction-api); docker-compose.yml chooses the command.
# Python 3.14 is the version the registered model was saved with.
FROM python:3.14-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
# The project folder (code, mlflow.db and mlruns) is mounted at run time, see docker-compose.yml.
ENV MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
