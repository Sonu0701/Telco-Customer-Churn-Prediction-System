#!/usr/bin/env python3
"""
Runs sequentially: load → validate → preprocess → feature engineering → train
"""

import os
import sys
import time
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score
)
from xgboost import XGBClassifier

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local modules
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data


def main(args):

    # MLflow setup
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run():

        mlflow.log_param("model", "xgboost")
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size)

        # LOAD DATA
        print("🔄 Loading data...")
        df = load_data(args.input)
        print(f"✅ Data loaded: {df.shape}")

        # VALIDATION
        print("🔍 Validating data...")
        try:
            is_valid, failed = validate_telco_data(df)
            if not is_valid:
                print(f"⚠️ Validation issues: {failed}")
            else:
                print("✅ Data validation passed")
        except Exception as e:
            print(f"⚠️ Validation skipped: {e}")

        # PREPROCESS
        print("🔧 Preprocessing...")
        df = preprocess_data(df)

        # FEATURE ENGINEERING
        print("🛠️ Building features...")
        target = args.target
        df_enc = build_features(df, target_col=target)

        # bool → int
        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)

        print(f"✅ Features ready: {df_enc.shape}")

        # SPLIT
        X = df_enc.drop(columns=[target])
        y = df_enc[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            stratify=y,
            random_state=42
        )

        # HANDLE IMBALANCE
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        # MODEL
        print("🤖 Training model...")
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=42,
            eval_metric="logloss"
        )

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        mlflow.log_metric("train_time", train_time)

        # EVALUATION
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= args.threshold).astype(int)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, proba)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        print(f"\n🎯 Precision: {precision:.3f}")
        print(f"🎯 Recall: {recall:.3f}")
        print(f"🎯 F1 Score: {f1:.3f}")
        print(f"🎯 ROC-AUC: {roc_auc:.3f}")

        # ✅ SAVE FEATURE COLUMNS (CRITICAL FIX)
        feature_cols = X.columns.tolist()
        feature_path = "feature_columns.txt"

        with open(feature_path, "w") as f:
            for col in feature_cols:
                f.write(col + "\n")

        mlflow.log_artifact(feature_path)

        # SAVE MODEL
        mlflow.sklearn.log_model(model, "model")

        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn")

    args = p.parse_args()
    main(args)