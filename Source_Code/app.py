
import json
from pathlib import Path

import joblib
import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
    send_from_directory,
    make_response,
)
from werkzeug.utils import secure_filename

from config import DATA_PATH, MODEL_DIR, OUTPUT_DIR
from mi_utils import load_dataset, train_models


app = Flask(__name__)
app.secret_key = "thyroid-dashboard-local-development-key"

ALLOWED_EXTENSIONS = {".csv"}

POSITIVE_LABELS = {
    "1",
    "yes",
    "true",
    "positive",
    "hypothyroid",
    "hypothyroidism",
}


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def current_bundle():
    """Load the trained model bundle, if available."""
    model_path = MODEL_DIR / "random_forest.joblib"

    if not model_path.exists():
        return None

    return joblib.load(model_path)


def load_summary():
    """Load saved model evaluation results."""
    results_path = OUTPUT_DIR / "evaluation_results.json"

    if not results_path.exists():
        return None

    try:
        return json.loads(
            results_path.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError):
        app.logger.exception("Could not load evaluation results")
        return None


def dataset_exists():
    return Path(DATA_PATH).exists()


def charts():
    """Return generated chart filenames."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        path.name for path in OUTPUT_DIR.glob("*.png")
    )


def positive_class_index(classes, probabilities=None):
    """Find a recognized positive label or fall back to the most likely class."""
    for index, label in enumerate(classes):
        if str(label).strip().lower() in POSITIVE_LABELS:
            return index

    if probabilities is not None and len(probabilities):
        return int(probabilities.argmax())

    return None


def prepare_features(frame, bundle):
    """Convert incoming feature data to the types used during training."""
    frame = frame.copy()

    pipeline = bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]

    for transformer_name, _, columns in preprocessor.transformers_:
        if transformer_name == "numeric":
            for column in columns:
                frame[column] = pd.to_numeric(
                    frame[column], errors="coerce"
                )

        elif transformer_name == "categorical":
            for column in columns:
                frame[column] = frame[column].map(
                    lambda value: (
                        str(value).strip()
                        if pd.notna(value)
                        else None
                    )
                )

    return frame


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.route("/")
def home():
    summary = None

    if dataset_exists():
        try:
            _, _, summary = load_dataset()
        except Exception as exc:
            app.logger.exception("Could not load dataset on home page")
            flash(f"Dataset issue: {exc}", "error")

    trained = (
        MODEL_DIR / "random_forest.joblib"
    ).exists()

    return render_template(
        "home.html",
        summary=summary,
        trained=trained,
    )


# --------------------------------------------------
# UPLOAD DATASET
# --------------------------------------------------

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        uploaded_file = request.files.get("dataset")

        if uploaded_file is None or not uploaded_file.filename:
            flash("Please choose a CSV file.", "error")
            return redirect(url_for("upload"))

        # Normalize Windows-style paths and sanitize the filename.
        original_filename = uploaded_file.filename.replace("\\", "/")
        original_filename = original_filename.rsplit("/", 1)[-1]
        safe_filename = secure_filename(original_filename)

        # Case-insensitive extension validation.
        if Path(safe_filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            flash(
                f"Invalid file '{original_filename}'. "
                "Please select a file ending in .csv.",
                "error",
            )
            return redirect(url_for("upload"))

        try:
            # Check that the uploaded file is readable as CSV.
            uploaded_file.stream.seek(0)

            preview = pd.read_csv(
                uploaded_file.stream,
                nrows=5,
            )

            uploaded_file.stream.seek(0)

            if len(preview.columns) < 2:
                raise ValueError(
                    "The CSV must contain at least one feature "
                    "column and one target column."
                )

            # Save to the same location used by config.py.
            Path(DATA_PATH).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            uploaded_file.save(DATA_PATH)

            # Remove stale models and evaluation results.
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            for path in MODEL_DIR.glob("*.joblib"):
                path.unlink(missing_ok=True)

            for path in OUTPUT_DIR.iterdir():
                if path.is_file():
                    path.unlink(missing_ok=True)

            # Validate the saved dataset using the project utility.
            _, _, dataset_summary = load_dataset()

            flash(
                f"Dataset uploaded successfully: "
                f"{dataset_summary['rows']:,} rows and "
                f"{dataset_summary['features']} features.",
                "success",
            )

            return redirect(url_for("analysis"))

        except Exception as exc:
            app.logger.exception("Dataset upload failed")

            flash(
                f"Upload or dataset validation failed: {exc}",
                "error",
            )

            return redirect(url_for("upload"))

    return render_template(
        "upload.html",
        dataset_exists=dataset_exists(),
    )


# --------------------------------------------------
# DATASET ANALYSIS
# --------------------------------------------------

@app.route("/analysis")
def analysis():
    summary = None
    error = None

    if not dataset_exists():
        error = "Upload a dataset before opening Analysis."
    else:
        try:
            _, _, summary = load_dataset()
        except Exception as exc:
            app.logger.exception("Dataset analysis failed")
            error = str(exc)

    return render_template(
        "analysis.html",
        summary=summary,
        error=error,
        image_files=charts(),
    )


# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------

@app.route("/train", methods=["GET", "POST"])
def train():
    summary = load_summary()
    error = None

    if request.method == "POST":
        try:
            if not dataset_exists():
                raise ValueError("Upload a dataset before training.")

            summary = train_models()

            flash(
                "Model training completed successfully.",
                "success",
            )

            return redirect(url_for("results"))

        except Exception as exc:
            app.logger.exception("Model training failed")
            error = str(exc)

    return render_template(
        "train.html",
        summary=summary,
        error=error,
        dataset_exists=dataset_exists(),
    )


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

@app.route("/results")
def results():
    return render_template(
        "results.html",
        summary=load_summary(),
        image_files=charts(),
    )


# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

@app.route("/compare")
def compare():
    return render_template(
        "compare.html",
        summary=load_summary(),
        image_files=charts(),
    )


# --------------------------------------------------
# SINGLE PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["GET", "POST"])
def predict():
    bundle = None
    prediction = None
    error = None
    feature_columns = []

    try:
        bundle = current_bundle()
    except Exception as exc:
        app.logger.exception("Could not load trained model")
        error = f"Could not load trained model: {exc}"

    if bundle:
        feature_columns = bundle["feature_columns"]

    if request.method == "POST":
        try:
            bundle = current_bundle()

            if not bundle:
                raise ValueError("Train the model before predicting.")

            feature_columns = bundle["feature_columns"]

            row = {
                column: request.form.get(column, "").strip() or None
                for column in feature_columns
            }

            frame = pd.DataFrame(
                [row],
                columns=feature_columns,
            )

            frame = prepare_features(frame, bundle)

            pipeline = bundle["pipeline"]
            predicted_label = pipeline.predict(frame)[0]
            probabilities = pipeline.predict_proba(frame)[0]
            classes = list(pipeline.classes_)

            predicted_index = classes.index(predicted_label)
            positive_index = positive_class_index(
                classes,
                probabilities,
            )

            prediction = {
                "label": str(predicted_label),
                "probability_percent": round(
                    float(probabilities[predicted_index]) * 100,
                    2,
                ),
                "positive_class": str(classes[positive_index]),
                "positive_probability_percent": round(
                    float(probabilities[positive_index]) * 100,
                    2,
                ),
            }

        except Exception as exc:
            app.logger.exception("Single prediction failed")
            error = str(exc)

    return render_template(
        "predict.html",
        feature_columns=feature_columns,
        prediction=prediction,
        error=error,
        has_model=bundle is not None,
    )


# --------------------------------------------------
# BATCH PREDICTION
# --------------------------------------------------

@app.route("/batch", methods=["GET", "POST"])
def batch():
    error = None
    result_csv = None
    count = None

    if request.method == "POST":
        try:
            bundle = current_bundle()

            if not bundle:
                raise ValueError("Train the model before batch prediction.")

            uploaded_file = request.files.get("batch_file")

            if (
                uploaded_file is None
                or not uploaded_file.filename
                or Path(
                    uploaded_file.filename.replace("\\", "/")
                    .rsplit("/", 1)[-1]
                ).suffix.lower() not in ALLOWED_EXTENSIONS
            ):
                raise ValueError("Please select a valid CSV file.")

            df = pd.read_csv(
                uploaded_file,
                na_values=["?", ""],
            )

            expected_columns = bundle["feature_columns"]

            missing_columns = [
                column
                for column in expected_columns
                if column not in df.columns
            ]

            if missing_columns:
                raise ValueError(
                    "The CSV is missing required feature columns: "
                    + ", ".join(missing_columns)
                )

            X = df[expected_columns].copy()
            X = prepare_features(X, bundle)

            pipeline = bundle["pipeline"]
            predicted_labels = pipeline.predict(X)
            probabilities = pipeline.predict_proba(X)
            classes = list(pipeline.classes_)

            positive_index = positive_class_index(
                classes,
                probabilities.mean(axis=0),
            )

            df["prediction"] = predicted_labels
            df["predicted_class_probability_percent"] = (
                probabilities.max(axis=1) * 100
            ).round(2)

            df["positive_class"] = str(classes[positive_index])
            df["positive_class_probability_percent"] = (
                probabilities[:, positive_index] * 100
            ).round(2)

            result_csv = df.to_csv(index=False)
            count = len(df)

        except Exception as exc:
            app.logger.exception("Batch prediction failed")
            error = str(exc)

    return render_template(
        "batch.html",
        error=error,
        result_csv=result_csv,
        count=count,
    )


# --------------------------------------------------
# REPORT
# --------------------------------------------------

@app.route("/report")
def report():
    summary = load_summary()

    if not summary:
        flash(
            "Train the model before generating a report.",
            "error",
        )
        return redirect(url_for("train"))

    return render_template(
        "report.html",
        summary=summary,
        image_files=charts(),
    )


@app.route("/download-report")
def download_report():
    summary = load_summary()

    if not summary:
        flash("Train the model before downloading the report.", "error")
        return redirect(url_for("train"))

    report_path = OUTPUT_DIR / "evaluation_results.json"

    return send_file(
        report_path,
        as_attachment=True,
        download_name="thyroid_evaluation_results.json",
    )


# --------------------------------------------------
# DOWNLOAD BATCH RESULTS
# --------------------------------------------------

@app.route("/download-batch", methods=["POST"])
def download_batch():
    result_csv = request.form.get("result_csv", "")

    if not result_csv:
        flash("No batch prediction results are available.", "error")
        return redirect(url_for("batch"))

    response = make_response(result_csv)
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = (
        "attachment; filename=thyroid_predictions.csv"
    )

    return response


# --------------------------------------------------
# CHARTS
# --------------------------------------------------

@app.route("/charts/<path:filename>")
def chart_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/health")
def health():
    return {
        "status": "ok",
        "dataset_loaded": dataset_exists(),
        "model_trained": (
            MODEL_DIR / "random_forest.joblib"
        ).exists(),
    }


# --------------------------------------------------
# ERROR HANDLERS
# --------------------------------------------------

@app.errorhandler(404)
def page_not_found(error):
    return "Page not found.", 404


@app.errorhandler(500)
def internal_server_error(error):
    app.logger.error("Unhandled server error: %s", error)

    return (
        "Internal server error. Check the VS Code terminal "
        "for the full Python traceback.",
        500,
    )


# --------------------------------------------------
# START APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )