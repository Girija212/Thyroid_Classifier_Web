# Thyroid Classifier — AI-ML-037

A Flask dashboard for thyroid/hypothyroid classification with CSV upload, exploratory analysis, Random Forest and Logistic Regression training, model comparison, single-patient prediction, batch prediction, and report download.

## Quick start (Windows / VS Code)

1. Extract the ZIP.
2. Open `Source_Code` in VS Code.
3. Open Terminal and run:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

4. Open `http://127.0.0.1:5000`.
5. Open **Upload** and upload your `hypothyroid.csv`. The file must contain a target/diagnosis column as well as feature columns.
6. Open **Train** and click **Start Training**. Then visit **Results** and **Predict**.

## Dataset expectations

The app is designed for a thyroid dataset with approximately 3,772 rows and a target indicating hypothyroid/positive versus negative/normal. Common target column names supported include `Class`, `target`, `diagnosis`, `binaryClass`, `Hypothyroid`, and `label`. The app tries to infer a positive class from values such as `hypothyroid`, `positive`, `P`, `yes`, or `1`. Review the target distribution after upload.

CSV missing values such as `?`, blank, `NA`, and `null` are treated as missing. Categorical fields are imputed and one-hot encoded. Numeric fields are median-imputed and scaled. SMOTE is applied only inside each training pipeline.

## Important

- Do not use model output as a medical diagnosis.
- Performance metrics are calculated from the dataset actually uploaded and may differ from the reference video.
- Do not claim the reference video's metrics unless your own run reproduces them.
- The dashboard is a local educational prototype, not a production medical application.
