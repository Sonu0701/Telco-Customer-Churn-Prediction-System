import os
import pandas as pd
import mlflow
import glob

# === MODEL LOADING ===
MODEL_DIR = "/app/model"

try:
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"✅ Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"❌ Failed to load model from {MODEL_DIR}: {e}")

    # Fallback: load latest model from mlruns
    local_model_paths = glob.glob("./mlruns/*/*/artifacts/model")
    if not local_model_paths:
        raise Exception("❌ No model found in mlruns")

    latest_model = max(local_model_paths, key=os.path.getmtime)
    model = mlflow.pyfunc.load_model(latest_model)
    MODEL_DIR = latest_model
    print(f"✅ Fallback: Loaded model from {latest_model}")

# === FEATURE COLUMN LOADING (FIXED 🔥) ===
try:
    # First try inside model folder
    feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")

    # If not found → go one level up (correct MLflow structure)
    if not os.path.exists(feature_file):
        feature_file = os.path.join(os.path.dirname(MODEL_DIR), "feature_columns.txt")

    # If still not found → use root level file
    if not os.path.exists(feature_file):
        feature_file = "./feature_columns.txt"

    # If still not found → use /app root
    if not os.path.exists(feature_file):
        feature_file = "/app/feature_columns.txt"

    with open(feature_file) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]

    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from {feature_file}")

except Exception as e:
    raise Exception(f"❌ Failed to load feature columns: {e}")

# === CONSTANTS ===
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# === TRANSFORMATION ===
def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # numeric
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # binary
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                .str.strip()
                .map(mapping)
                .fillna(0)
                .astype(int)
            )

    # one-hot
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # bool → int
    for c in df.select_dtypes(include=["bool"]).columns:
        df[c] = df[c].astype(int)

    # align with training schema
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    return df

# === PREDICT ===
def predict(input_dict: dict) -> str:
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    try:
        pred = model.predict(df_enc)

        if hasattr(pred, "tolist"):
            pred = pred.tolist()

        result = pred[0] if isinstance(pred, list) else pred

    except Exception as e:
        raise Exception(f"Prediction failed: {e}")

    return "Likely to churn" if result == 1 else "Not likely to churn"