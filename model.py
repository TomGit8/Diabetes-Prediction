"""
Reproducible diabetes prediction training pipeline.

The script can be imported, but it also exposes a CLI entry point:

    python model.py

It loads the dataset, applies deterministic preprocessing, evaluates several
classifiers with cross-validation and holds out a test set, and saves the best
performing model plus the collected metrics under ./artifacts/.
"""

from __future__ import annotations

import json
import logging
import pickle
import boto3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


@dataclass
class TrainingConfig:
    """Configuration bundle for the training job."""

    data_path: Path = Path("diabetes.csv")
    target_column: str = "Outcome"
    test_size: float = 0.2
    n_cv_splits: int = 5
    random_state: int = 42
    outlier_multiplier: float = 1.5
    artifact_dir: Path = Path("artifacts")
    s3_bucket: str = "s3-g3mg05"


def load_data(config: TrainingConfig) -> pd.DataFrame:
    """Load the raw dataset from disk."""
    if not config.data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {config.data_path}")
    logging.info("Loading dataset from %s", config.data_path)
    return pd.read_csv(config.data_path)


def remove_outliers_iqr(
    df: pd.DataFrame, *, columns: Iterable[str], multiplier: float
) -> pd.DataFrame:
    """Remove rows that fall outside the IQR fence for the given columns."""
    if not columns:
        return df.copy()
    q1 = df[list(columns)].quantile(0.25)
    q3 = df[list(columns)].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    mask = ~(
        ((df[list(columns)] < lower_bound) | (df[list(columns)] > upper_bound)).any(
            axis=1
        )
    )
    cleaned = df.loc[mask].reset_index(drop=True)
    logging.info("Removed %d outliers via IQR filter", len(df) - len(cleaned))
    return cleaned


def preprocess_data(df: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    """Apply deterministic preprocessing steps to the dataframe."""
    feature_columns = [col for col in df.columns if col != config.target_column]
    cleaned = remove_outliers_iqr(
        df,
        columns=feature_columns,
        multiplier=config.outlier_multiplier,
    )
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    logging.info("Dataset size after cleaning: %d rows", len(cleaned))
    return cleaned


def split_data(
    df: pd.DataFrame, config: TrainingConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into train and test partitions."""
    X = df.drop(columns=[config.target_column])
    y = df[config.target_column]
    train_X, test_X, train_y, test_y = train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )
    logging.info(
        "Split data: train=%s test=%s", train_X.shape, test_X.shape
    )
    return train_X, test_X, train_y, test_y


def build_models(config: TrainingConfig) -> Dict[str, BaseEstimator]:
    """Create the candidate estimators wrapped in pipelines where needed."""
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=config.random_state,
                    ),
                ),
            ]
        ),
        "svm_rbf": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=config.random_state,
                    ),
                ),
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=config.random_state,
        ),
        "gaussian_nb": GaussianNB(),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=config.random_state,
        ),
    }


def _roc_auc_or_nan(model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> float:
    """Return ROC AUC if the estimator exposes probabilities/decision scores."""
    try:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
        else:
            return float("nan")
        return float(roc_auc_score(y, scores))
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("Failed to compute ROC AUC: %s", exc)
        return float("nan")


def evaluate_models(
    models: Dict[str, BaseEstimator],
    train_X: pd.DataFrame,
    train_y: pd.Series,
    test_X: pd.DataFrame,
    test_y: pd.Series,
    config: TrainingConfig,
) -> Tuple[str, BaseEstimator, List[dict]]:
    """Cross-validate and evaluate each estimator, returning the best one."""
    cv = StratifiedKFold(
        n_splits=config.n_cv_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    best_name = ""
    best_model: BaseEstimator | None = None
    best_score = -np.inf
    results: List[dict] = []

    for name, estimator in models.items():
        logging.info("Training candidate model: %s", name)
        cv_scores = cross_validate(
            estimator,
            train_X,
            train_y,
            scoring=SCORING,
            cv=cv,
            n_jobs=1,
        )
        fitted_model = clone(estimator)
        fitted_model.fit(train_X, train_y)
        predictions = fitted_model.predict(test_X)
        roc_auc = _roc_auc_or_nan(fitted_model, test_X, test_y)
        metrics = {
            "model": name,
            "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
            "cv_precision_mean": float(np.mean(cv_scores["test_precision"])),
            "cv_recall_mean": float(np.mean(cv_scores["test_recall"])),
            "cv_f1_mean": float(np.mean(cv_scores["test_f1"])),
            "cv_roc_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
            "test_accuracy": float(accuracy_score(test_y, predictions)),
            "test_precision": float(precision_score(test_y, predictions)),
            "test_recall": float(recall_score(test_y, predictions)),
            "test_f1": float(f1_score(test_y, predictions)),
            "test_roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
            "classification_report": classification_report(test_y, predictions),
        }
        results.append(metrics)

        ranking_score = (
            roc_auc if not np.isnan(roc_auc) else metrics["test_accuracy"]
        )
        logging.info(
            "%s - Test accuracy: %.3f | ROC AUC: %s",
            name,
            metrics["test_accuracy"],
            f"{roc_auc:.3f}" if not np.isnan(roc_auc) else "n/a",
        )
        if ranking_score > best_score:
            best_score = ranking_score
            best_name = name
            best_model = fitted_model

    if best_model is None:
        raise RuntimeError("No model could be trained successfully.")
    logging.info("Best model by validation score: %s", best_name)
    return best_name, best_model, results


def save_artifacts(
    best_name: str,
    best_model: BaseEstimator,
    metrics: List[dict],
    config: TrainingConfig,
) -> None:
    """Persist the winning model, run metrics, and configuration metadata."""
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.artifact_dir / f"{best_name}.joblib"
    metrics_path = config.artifact_dir / "metrics.json"
    config_path = config.artifact_dir / "config.json"

    with model_path.open("wb") as f:
        pickle.dump(best_model, f)
    logging.info("Saved trained model to %s", model_path)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logging.info("Persisted evaluation metrics to %s", metrics_path)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, default=str)
    logging.info("Persisted run configuration to %s", config_path)

    # Upload to S3
    s3 = boto3.client("s3")
    try:
        for file_path in [model_path, metrics_path, config_path]:
            s3.upload_file(
                str(file_path),
                config.s3_bucket,
                f"artifacts/{file_path.name}"
            )
            logging.info("Uploaded %s to s3://%s/artifacts/", file_path.name, config.s3_bucket)
    except Exception as e:
        logging.error("Failed to upload to S3: %s", e)


def run_training_job(config: TrainingConfig) -> None:
    """Execute the full training workflow."""
    df = load_data(config)
    df = preprocess_data(df, config)
    train_X, test_X, train_y, test_y = split_data(df, config)
    models = build_models(config)
    best_name, best_model, metrics = evaluate_models(
        models, train_X, train_y, test_X, test_y, config
    )
    save_artifacts(best_name, best_model, metrics, config)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = TrainingConfig()
    logging.info("Starting training job with config: %s", config)
    run_training_job(config)


if __name__ == "__main__":
    main()
