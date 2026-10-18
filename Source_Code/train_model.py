
# train_model.py

import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)

from mi_utils import clean_thyroid_dataframe


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "thyroid_data.csv"
TARGET_COL = "Class"
MODEL_PATH = "models/thyroid_model.pkl"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}. "
        "Place thyroid_data.csv in your project folder."
    )

print("Loading thyroid dataset...")

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# ============================================================
# 2. CLEAN DATA
# ============================================================

print("\nCleaning dataset...")

df = clean_thyroid_dataframe(df)

# Find target column without case sensitivity.
target_matches = [
    col for col in df.columns
    if str(col).strip().lower() == TARGET_COL.lower()
]

if not target_matches:
    raise ValueError(
        f"Target column '{TARGET_COL}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

target_col = target_matches[0]

# Remove rows with missing target values.
df = df.dropna(subset=[target_col]).copy()

# Remove columns that contain no useful values.
df = df.dropna(axis=1, how="all")

if target_col not in df.columns:
    raise ValueError("Target column was removed during cleaning.")

print("Cleaned dataset shape:", df.shape)
print("\nTarget class distribution:")
print(df[target_col].value_counts())


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

if X.empty:
    raise ValueError("No feature columns found in the dataset.")

if y.nunique() < 2:
    raise ValueError(
        "The target must contain at least two different classes."
    )

# Convert numeric-looking feature columns to numeric values.
# Invalid numeric strings become NaN and are imputed later.
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        continue

    converted = pd.to_numeric(X[col], errors="coerce")

    # Convert only when every non-missing value is numeric.
    if converted.notna().sum() == X[col].notna().sum():
        X[col] = converted

# Detect categorical and numeric features automatically.
categorical_cols = X.select_dtypes(
    include=["object", "string", "category", "bool"]
).columns.tolist()

numeric_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)


# ============================================================
# 4. BUILD PREPROCESSING PIPELINES
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])

transformers = []

if numeric_cols:
    transformers.append(
        ("numeric", numeric_pipeline, numeric_cols)
    )

if categorical_cols:
    # Ensure each categorical column uses a consistent type.
    for col in categorical_cols:
        X[col] = X[col].map(
            lambda value: str(value).strip()
            if pd.notna(value) else float("nan")
        )

    transformers.append(
        ("categorical", categorical_pipeline, categorical_cols)
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)


# ============================================================
# 5. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

print("\nSplitting dataset...")

# Stratification preserves class proportions when possible.
class_counts = y.value_counts()

if class_counts.min() < 2:
    raise ValueError(
        "At least one target class has fewer than 2 records. "
        "More examples are needed for a stratified train/test split."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 6. CREATE RANDOM FOREST MODEL
# ============================================================

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    )
])


# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\nTraining Random Forest classifier...")

model.fit(X_train, y_train)

print("Training completed successfully.")


# ============================================================
# 8. PREDICT ON TEST DATA
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)


# ============================================================
# 9. EVALUATE MODEL
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

print("\n" + "=" * 55)
print("       THYROID CLASSIFICATION RESULTS")
print("=" * 55)

print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(
    f"Balanced accuracy: "
    f"{balanced_accuracy:.4f} ({balanced_accuracy * 100:.2f}%)"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 10. SAVE TRAINED MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\n" + "=" * 55)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 55)
print(f"Saved location: {MODEL_PATH}")

print("\nAll steps completed successfully.")