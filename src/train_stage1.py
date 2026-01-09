from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import argparse
import pandas as pd

from src.preprocessing import build_preprocessor
from xgboost import XGBClassifier
from src.target_builder import create_targets


def train_stage1(df, preprocessor, model):
    """
    Stage-1:
    Predict disease CATEGORY (Bacterial, Viral, etc.)
    """

    # ==============================
    # 1️⃣ TARGET & FEATURES
    # ==============================
    y = df["target_category"]
    X = df.drop(columns=["target_category", "target_disease"], errors="ignore")

    # ==============================
    # 2️⃣ LABEL ENCODING (MANDATORY)
    # ==============================
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # ==============================
    # 3️⃣ TRAIN-TEST SPLIT
    # ==============================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # ==============================
    # 4️⃣ PIPELINE
    # ==============================
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # ==============================
    # 5️⃣ TRAIN
    # ==============================
    pipeline.fit(X_train, y_train)

    # ==============================
    # 6️⃣ RETURN EVERYTHING NEEDED
    # ==============================
    return pipeline, X_test, y_test, label_encoder


def main(args=None):
    parser = argparse.ArgumentParser(description="Train and save stage-1 pipeline")
    parser.add_argument("--data", default="data/full_animal_disease_dataset.csv", help="Path to CSV dataset")
    parser.add_argument("--models-dir", default="models", help="Directory to save pipeline and encoder")
    parser.add_argument("--sample-frac", type=float, default=0.2, help="Fraction of data to sample for quick runs (0<frac<=1)")
    parser.add_argument("--n-estimators", type=int, default=100, help="XGB n_estimators")
    parsed = parser.parse_args(args=args)

    df = pd.read_csv(parsed.data)
    # Build target columns from one-hot disease columns
    df = create_targets(df)
    if parsed.sample_frac and 0 < parsed.sample_frac < 1:
        df = df.sample(frac=parsed.sample_frac, random_state=42)

    preprocessor = build_preprocessor(df)
    model = XGBClassifier(n_estimators=parsed.n_estimators, use_label_encoder=False, eval_metric='logloss', random_state=42)

    pipeline, X_test, y_test, label_encoder = train_stage1(df, preprocessor, model)

    os.makedirs(parsed.models_dir, exist_ok=True)
    pipeline_path = os.path.join(parsed.models_dir, "stage1_pipeline.pkl")
    encoder_path = os.path.join(parsed.models_dir, "category_encoder.pkl")

    joblib.dump(pipeline, pipeline_path)
    joblib.dump(label_encoder, encoder_path)

    print(f"Saved stage1 pipeline -> {pipeline_path}")
    print(f"Saved category encoder -> {encoder_path}")


if __name__ == '__main__':
    main()
