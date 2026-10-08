"""MLflow tracking and model registry helpers (used by run_pipeline, not a filter)."""
import mlflow
from mlflow import MlflowClient

import config
from pipeline.evaluate import targets_met


def _setup() -> None:
    """Point MLflow at the local database and create the experiment once."""
    mlflow.set_tracking_uri(config.TRACKING_URI)
    # Create the experiment with a fixed artifact folder the first time only.
    if MlflowClient().get_experiment_by_name(config.EXPERIMENT) is None:
        mlflow.create_experiment(config.EXPERIMENT, artifact_location=config.ARTIFACT_ROOT)
    mlflow.set_experiment(config.EXPERIMENT)


def log_candidate(name: str, model, X_example, threshold: float, val_metrics: dict,
                  test_metrics: dict | None = None, register: bool = False,
                  allow_failed_acceptance: bool = False):
    """Log one candidate as an MLflow run; register it when it is the winner."""
    accepted = (test_metrics is not None
                and {"recall", "fpr", "loss_reduction"}.issubset(test_metrics)
                and all(targets_met(test_metrics).values()))
    if register and not accepted and not allow_failed_acceptance:
        raise ValueError("Cannot promote a model with missing or failed acceptance checks. "
                         "Use allow_failed_acceptance=True only for a demonstration.")
    _setup()
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({
            "model_quality_acceptance": "not_evaluated" if test_metrics is None else
                                        ("passed" if accepted else "failed"),
            "deployment_scope": "not_registered" if not register else
                                ("demonstration_only" if allow_failed_acceptance else "quality_gated")})
        # Parameters: what was used. The API later reads "threshold" from here (BR-002).
        mlflow.log_param("model_name", name)
        mlflow.log_param("seed", config.SEED)
        mlflow.log_param("threshold", threshold)
        mlflow.log_params({"validation_recall_target": config.VALIDATION_RECALL_TARGET,
                           "minimum_validation_recall": config.TARGET_RECALL,
                           "max_validation_fpr": config.MAX_FPR,
                           "minimum_acceptance_recall": config.MIN_ACCEPTANCE_RECALL,
                           "desired_recall_goal": config.TARGET_RECALL,
                           "acceptance_policy_revision": config.RECALL_POLICY_REVISION})
        mlflow.log_params({f"clf__{key}": value for key, value in model.named_steps["clf"].get_params().items()})
        # Metrics: what came out (counts such as tn/fp are left out of the metric table).
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()
                            if k not in ("tn", "fp", "fn", "tp")})
        if test_metrics is not None:  # only the winner is evaluated on the test set
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()
                                if k not in ("tn", "fp", "fn", "tp")})
        # cloudpickle is used because the default format refuses tree-based models.
        info = mlflow.sklearn.log_model(
            sk_model=model, name="model", input_example=X_example.head(2),
            serialization_format="cloudpickle",
            registered_model_name=config.MODEL_NAME if register else None)
    version = None
    if register:
        # Move the "champion" alias to the new version; the API loads whatever it points to.
        version = info.registered_model_version
        MlflowClient().set_registered_model_alias(config.MODEL_NAME, config.ALIAS, version)
    return run.info.run_id, version
