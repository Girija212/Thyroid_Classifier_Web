"""Utilities required by the Thyroid Classifier Dashboard app.py."""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DATA_PATH, MODEL_DIR, OUTPUT_DIR


# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

MISSING = {
    "", "?", "na", "n/a", "nan", "null", "none",
    "missing", "-", "--"
}

YES = {"yes", "y", "true", "t", "1", "1.0"}
NO = {"no", "n", "false", "f", "0", "0.0"}

SEX = {
    "m": "Male",
    "male": "Male",
    "man": "Male",
    "f": "Female",
    "female": "Female",
    "woman": "Female",
}

BINARY = {
    "on_thyroxine",
    "query_on_thyroxine",
    "on_antithyroid_medication",
    "sick",
    "pregnant",
    "thyroid_surgery",
    "i131_treatment",
    "query_hypothyroid",
    "query_hyperthyroid",
    "lithium",
    "goitre",
    "tumor",
    "hypopituitary",
    "psych",
}

TARGET_NAMES = {
    "class",
    "target",
    "diagnosis",
    "label",
    "thyroid",
    "binaryclass",
    "hypothyroid",
}

POSITIVE_LABELS = {
    "1",
    "yes",
    "true",
    "positive",
    "hypothyroid",
    "hypothyroidism",
    "p",
}


# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

def clean_thyroid_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Clean column names, missing markers and feature values."""

    if data is None or data.empty:
        raise ValueError("The uploaded CSV is empty.")

    df = data.copy()

    # Normalize column names.
    df.columns = [
        re.sub(r"\s+", "_", str(column).strip()).replace("-", "_")
        for column in df.columns
    ]

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            f"Duplicate column names found: {duplicates}"
        )

    # Convert common missing markers to NaN.
    for column in df.columns:
        if (
            df[column].dtype == "object"
            or pd.api.types.is_string_dtype(df[column])
        ):
            df[column] = df[column].map(
                lambda value: (
                    np.nan
                    if pd.isna(value)
                    or (
                        isinstance(value, str)
                        and value.strip().lower() in MISSING
                    )
                    else (
                        value.strip()
                        if isinstance(value, str)
                        else value
                    )
                )
            )

    # Normalize the sex column.
    column_lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    sex_column = column_lookup.get("sex")

    if sex_column is not None:
        df[sex_column] = df[sex_column].map(
            lambda value: (
                SEX.get(value.lower().strip(), value.strip())
                if isinstance(value, str)
                else value
            )
        )

    # Convert yes/no fields to 1/0.
    for column in df.columns:
        if str(column).strip().lower() not in BINARY:
            continue

        def convert_binary(value):
            if pd.isna(value):
                return np.nan

            if isinstance(value, (bool, np.bool_)):
                return int(value)

            if isinstance(value, (int, float, np.number)):
                if value == 1:
                    return 1
                if value == 0:
                    return 0
                return value

            text = str(value).strip().lower()

            if text in YES:
                return 1

            if text in NO:
                return 0

            return np.nan

        df[column] = df[column].map(convert_binary)

    # Convert numeric-looking feature columns.
    # Keep target and sex columns unchanged.
    for column in df.columns:
        column_name = str(column).strip().lower()

        if column == sex_column or column_name in TARGET_NAMES:
            continue

        if (
            df[column].dtype == "object"
            or pd.api.types.is_string_dtype(df[column])
        ):
            converted = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            original_count = df[column].notna().sum()

            if (
                original_count > 0
                and converted.notna().sum() == original_count
            ):
                df[column] = converted

    # Remove columns containing only missing values.
    return df.dropna(axis=1, how="all")


# --------------------------------------------------
# TARGET COLUMN DETECTION
# --------------------------------------------------

def _find_target(df: pd.DataFrame) -> str:
    """Find the target column using common thyroid column names."""

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    candidates = (
        "class",
        "target",
        "diagnosis",
        "binaryclass",
        "hypothyroid",
        "label",
        "thyroid",
    )

    for name in candidates:
        if name in lookup:
            return lookup[name]

    if len(df.columns) < 2:
        raise ValueError(
            "Dataset must contain feature columns and a target column."
        )

    # Fallback: use the last column as the target.
    return df.columns[-1]


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_dataset():
    """Return X, y and a summary dictionary for app.py."""

    path = Path(DATA_PATH)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Upload a CSV or update DATA_PATH in config.py."
        )

    try:
        df = pd.read_csv(
            path,
            na_values=["?", ""],
            low_memory=False,
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV file contains no data.") from exc
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            encoding="latin-1",
            na_values=["?", ""],
            low_memory=False,
        )

    if df.empty:
        raise ValueError("The CSV file contains no records.")

    df = clean_thyroid_dataframe(df)

    if len(df.columns) < 2:
        raise ValueError(
            "The dataset must contain at least one feature "
            "column and one target column."
        )

    target = _find_target(df)

    # Remove records without target values.
    df = df.dropna(subset=[target]).copy()

    if df.empty:
        raise ValueError(
            f"No records have a valid target in '{target}'."
        )

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    if X.empty:
        raise ValueError("No feature columns found.")

    # Normalize target labels without changing their meaning.
    y = y.map(
        lambda value: (
            value.strip()
            if isinstance(value, str)
            else value
        )
    )

    if y.nunique() < 2:
        raise ValueError(
            "Target column must contain at least two classes."
        )

    # Count recognized positive labels.
    positive = int(
        y.astype(str)
        .str.strip()
        .str.lower()
        .isin(POSITIVE_LABELS)
        .sum()
    )

    class_counts = y.value_counts()

    # IMPORTANT:
    # positive_rate is a decimal between 0 and 1.
    # analysis.html multiplies it by 100 for display.
    summary = {
        "rows": int(len(df)),
        "features": int(X.shape[1]),
        "positive": positive,
        "positive_rate": (
            float(positive / len(df))
            if len(df) > 0
            else 0.0
        ),
        "target": str(target),
        "classes": {
            str(label): int(count)
            for label, count in class_counts.items()
        },
        "missing_values": int(X.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": [str(column) for column in df.columns],
    }

    return X, y, summary


# --------------------------------------------------
# MODEL PIPELINE
# --------------------------------------------------

def _make_pipeline(X: pd.DataFrame) -> Pipeline:
    """Create preprocessing and Random Forest classification steps."""

    categorical_columns = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    transformers = []

    # Numeric preprocessing.
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            )
        )

    # Categorical preprocessing.
    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError(
            "No usable numeric or categorical feature columns were found."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


# --------------------------------------------------
# MODEL TRAINING AND EVALUATION
# --------------------------------------------------

def train_models():
    """Train the model and save evaluation results and charts."""

    X, y, dataset_summary = load_dataset()

    class_counts = y.value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "Every target class needs at least two records "
            "for a stratified train/test split."
        )

    # Split dataset into training and testing records.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Build and train pipeline.
    pipeline = _make_pipeline(X)
    pipeline.fit(X_train, y_train)

    # Generate predictions.
    predictions = pipeline.predict(X_test)
    classes = list(pipeline.classes_)

    # Evaluate model.
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=classes,
    )

    accuracy = float(
        accuracy_score(y_test, predictions)
    )

    balanced_accuracy = float(
        balanced_accuracy_score(y_test, predictions)
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    # Create required directories.
    model_dir = Path(MODEL_DIR)
    output_dir = Path(OUTPUT_DIR)

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale charts before creating new charts.
    for old_chart in output_dir.glob("*.png"):
        old_chart.unlink(missing_ok=True)

    # Save model bundle expected by app.py.
    bundle = {
        "pipeline": pipeline,
        "feature_columns": list(X.columns),
        "target_column": dataset_summary["target"],
        "classes": classes,
    }

    joblib.dump(
        bundle,
        model_dir / "random_forest.joblib",
    )

    # Save evaluation summary. The results page expects best_model and results.
    weighted_f1 = float(
        report.get("weighted avg", {}).get("f1-score", 0.0)
    )

    summary = {
        "model": "Random Forest",
        "best_model": "Random Forest",
        "results": {
            "Random Forest": {
                "accuracy": accuracy,
                "f1": weighted_f1,
            }
        },
        "rows": dataset_summary["rows"],
        "features": dataset_summary["features"],
        "target": dataset_summary["target"],
        "classes": [str(label) for label in classes],
        "accuracy": accuracy,
        "accuracy_percent": round(accuracy * 100, 2),
        "balanced_accuracy": balanced_accuracy,
        "balanced_accuracy_percent": round(
            balanced_accuracy * 100,
            2,
        ),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_distribution": dataset_summary["classes"],
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": [str(label) for label in classes],
        "dataset": dataset_summary,
    }

    results_path = output_dir / "evaluation_results.json"

    results_path.write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # Generate confusion matrix chart.
    figure, axis = plt.subplots(figsize=(7, 6))

    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[str(label) for label in classes],
    ).plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
    )

    axis.set_title("Random Forest - Confusion Matrix")

    figure.tight_layout()
    figure.savefig(
        output_dir / "confusion_matrix.png",
        dpi=140,
        bbox_inches="tight",
    )

    plt.close(figure)

    # Generate feature importance chart.
    try:
        classifier = pipeline.named_steps["classifier"]

        feature_names = (
            pipeline.named_steps["preprocessor"]
            .get_feature_names_out()
        )

        importances = classifier.feature_importances_
        top_indices = np.argsort(importances)[-15:]

        figure, axis = plt.subplots(figsize=(9, 6))

        axis.barh(
            [str(feature_names[index]) for index in top_indices],
            importances[top_indices],
        )

        axis.set_title("Top Feature Importances")
        axis.set_xlabel("Importance")

        figure.tight_layout()
        figure.savefig(
            output_dir / "feature_importance.png",
            dpi=140,
            bbox_inches="tight",
        )

        plt.close(figure)

    except Exception:
        # Feature importance visualization is optional.
        plt.close("all")

    return summary
