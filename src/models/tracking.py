"""
MLflow logging helper module for Phase 2.
Standardizes experiment tagging, parameter tracking, feature list hashing,
and metric logging across all tasks: 'dropout', 'score', 'grade_band', 'ablation'.
"""
import hashlib
import json
import os
import subprocess
from typing import Dict, Any, List, Optional
import mlflow
import yaml


def setup_mlflow(tracking_uri: str = "sqlite:///mlruns.db"):
    """Initialize MLflow tracking URI."""
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    mlflow.set_tracking_uri(tracking_uri)


def get_git_commit() -> str:
    """Safely obtain git commit hash if available."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return commit
    except Exception:
        return "git-unavailable"


def get_feature_hash(feature_list: List[str]) -> str:
    """Compute deterministic SHA256 hash of feature names."""
    feat_str = ",".join(sorted(feature_list))
    return hashlib.sha256(feat_str.encode("utf-8")).hexdigest()[:12]


def log_model_experiment(
    experiment_name: str,
    run_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    feature_list: List[str],
    artifacts: Optional[Dict[str, str]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """
    Log a standardized run to MLflow under the specified experiment.
    """
    setup_mlflow()
    mlflow.set_experiment(experiment_name)

    all_tags = {
        "git_commit": get_git_commit(),
        "feature_hash": get_feature_hash(feature_list),
        "n_features": str(len(feature_list)),
    }
    if tags:
        all_tags.update(tags)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(all_tags)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        if artifacts:
            for art_name, art_path in artifacts.items():
                if os.path.exists(art_path):
                    mlflow.log_artifact(art_path)

        return run.info.run_id
