"""Tests for the prediction API: BR-001, BR-007, BR-008 and BR-009."""
import csv

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import config


@pytest.fixture()
def client(monkeypatch, tiny_model, tmp_path):
    """Test client for the API with a small model, threshold 0.5, model version 7 and a temporary log file."""
    monkeypatch.setattr(api_main, "load_champion", lambda: (tiny_model, "7", 0.5))
    monkeypatch.setattr(config, "LOG_PATH", tmp_path / "log.csv")
    with TestClient(api_main.app) as c:
        yield c


# ---- BR-001 ----------------------------------------------------------------------------
def test_BR001_label_is_probability_against_threshold(client, valid_payload):
    """The label must equal (probability >= threshold) and the probability must be between 0 and 1."""
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["is_fraud"] == (body["probability"] >= body["threshold"])


def test_BR001_boundary_probability_equal_to_threshold_is_fraud(monkeypatch, tiny_model,
                                                                tmp_path, valid_payload):
    """A probability exactly equal to the threshold counts as fraud (boundary case)."""
    p = float(tiny_model.predict_proba(
        __import__("pandas").DataFrame([valid_payload], columns=config.FEATURES))[0, 1])
    monkeypatch.setattr(api_main, "load_champion", lambda: (tiny_model, "7", p))
    monkeypatch.setattr(config, "LOG_PATH", tmp_path / "log.csv")
    with TestClient(api_main.app) as c:
        assert c.post("/predict", json=valid_payload).json()["is_fraud"] is True


# ---- BR-007 ----------------------------------------------------------------------------
def test_BR007_missing_field_rejected(client, valid_payload):
    """A request with a missing feature must get HTTP 422."""
    del valid_payload["V5"]
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_BR007_negative_amount_rejected(client, valid_payload):
    """A negative Amount must get HTTP 422."""
    valid_payload["Amount"] = -1.0
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_BR007_wrong_type_rejected(client, valid_payload):
    """A text value where a number is expected must get HTTP 422."""
    valid_payload["V1"] = "abc"
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_BR007_extra_field_rejected(client, valid_payload):
    """An unknown field (for example a card number) must get HTTP 422."""
    valid_payload["card_number"] = "4111111111111111"
    assert client.post("/predict", json=valid_payload).status_code == 422


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_BR007_nan_and_inf_rejected(client, valid_payload, bad):
    """NaN and infinity must get HTTP 422 and must not be logged."""
    parts = [f'"{k}": {bad if k == "V2" else v}' for k, v in valid_payload.items()]
    r = client.post("/predict", content="{" + ", ".join(parts) + "}",
                    headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert not config.LOG_PATH.exists()


def test_BR007_invalid_input_is_not_logged(client, valid_payload):
    """A rejected request must not create a log row."""
    valid_payload["Amount"] = -5.0
    client.post("/predict", json=valid_payload)
    assert not config.LOG_PATH.exists()


# ---- BR-008 ----------------------------------------------------------------------------
def test_BR008_every_call_is_logged(client, valid_payload):
    """Each successful call writes one log row with the expected columns and no feature values."""
    for _ in range(3):
        assert client.post("/predict", json=valid_payload).status_code == 200
    rows = list(csv.DictReader(config.LOG_PATH.open()))
    assert len(rows) == 3
    assert set(rows[0]) == {"timestamp", "request_id", "probability", "is_fraud",
                            "latency_ms", "model_version"}
    assert rows[0]["model_version"] == "7"
    assert not any(k.startswith("V") for k in rows[0])


def test_BR008_log_failure_is_an_error_not_silent(monkeypatch, tiny_model, tmp_path, valid_payload):
    """If the log cannot be written the API must answer HTTP 500, not continue silently."""
    monkeypatch.setattr(api_main, "load_champion", lambda: (tiny_model, "7", 0.5))
    monkeypatch.setattr(config, "LOG_PATH", tmp_path)  # a directory cannot be opened as a file
    with TestClient(api_main.app) as c:
        r = c.post("/predict", json=valid_payload)
    assert r.status_code == 500 and "log failed" in r.json()["detail"]


# ---- BR-009 ----------------------------------------------------------------------------
def test_BR009_loads_only_the_champion_alias(monkeypatch):
    """The API must load models:/fraud-model@champion and take the threshold from that model's run."""
    seen = {}

    class FakeMV:
        version = 3
        run_id = "abc"

    class FakeRun:
        class data:
            params = {"threshold": "0.25"}

    class FakeClient:
        def get_model_version_by_alias(self, name, alias):
            seen["alias"] = (name, alias)
            return FakeMV()

        def get_run(self, run_id):
            return FakeRun()

    monkeypatch.setattr("mlflow.sklearn.load_model",
                        lambda uri: seen.setdefault("uri", uri) or "model")
    monkeypatch.setattr(api_main, "MlflowClient", FakeClient)
    _, version, threshold = api_main.load_champion()
    assert seen["uri"] == "models:/fraud-model@champion"
    assert seen["alias"] == ("fraud-model", "champion")
    assert (version, threshold) == ("3", 0.25)


def test_BR009_response_carries_model_version(client, valid_payload):
    """Responses and /health must show the model version being served."""
    assert client.post("/predict", json=valid_payload).json()["model_version"] == "7"
    assert client.get("/health").json()["model_version"] == "7"


def test_BR009_tracking_uri_can_point_to_registry_service(monkeypatch):
    """In Docker Compose the API must reach the registry service through MLFLOW_TRACKING_URI."""
    import importlib
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://registry:5000")
    assert importlib.reload(config).TRACKING_URI == "http://registry:5000"
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    assert importlib.reload(config).TRACKING_URI.startswith("sqlite:///")  # default: the local file
