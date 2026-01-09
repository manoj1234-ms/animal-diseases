import os
import argparse
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor
from src.target_builder import create_targets


def train_stage2(df, models_dir, n_estimators=100, min_samples=20):
    """Train one stage-2 model per category and save dict of pipelines and encoders."""
    # ensure targets
    df = create_targets(df)

    # disease encoders per category
    stage2_models = {}
    disease_encoders = {}

    for category in df['target_category'].unique():
        sub = df[df['target_category'] == category].reset_index(drop=True)
        if len(sub) < min_samples:
            print(f"Skipping category {category}: only {len(sub)} samples (<{min_samples})")
            continue

        # X and y for stage-2: target_disease
        y = sub['target_disease']
        X = sub.drop(columns=['target_disease', 'target_category'], errors='ignore')

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        preprocessor = build_preprocessor(sub)
        model = XGBClassifier(n_estimators=n_estimators, use_label_encoder=False, eval_metric='logloss', random_state=42)

        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        print(f"Training stage-2 pipeline for category '{category}' on {len(X)} samples...")
        pipeline.fit(X, y_enc)

        stage2_models[category] = pipeline
        disease_encoders[category] = le

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(stage2_models, os.path.join(models_dir, 'stage2_models.pkl'))
    joblib.dump(disease_encoders, os.path.join(models_dir, 'disease_encoders.pkl'))

    print(f"Saved stage2 models ({len(stage2_models)}) and disease encoders.")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/full_animal_disease_dataset.csv')
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('--sample-frac', type=float, default=1.0)
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--min-samples', type=int, default=20)
    parsed = parser.parse_args(args=args)

    df = pd.read_csv(parsed.data)
    if 0 < parsed.sample_frac < 1.0:
        df = df.sample(frac=parsed.sample_frac, random_state=42)

    train_stage2(df, parsed.models_dir, n_estimators=parsed.n_estimators, min_samples=parsed.min_samples)


if __name__ == '__main__':
    main()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

def train_stage2_models(df, preprocessor):

    stage2_models = {}
    disease_label_encoders = {}

    categories = df["target_category"].unique()

    for category in categories:
        df_cat = df[df["target_category"] == category]

        # ✅ CORRECT COLUMNS
        X = df_cat.drop(columns=["target_category", "target_disease"])
        y = df_cat["target_disease"]

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss"
        )

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X, y_encoded)

        stage2_models[category] = pipeline
        disease_label_encoders[category] = label_encoder

    return stage2_models, disease_label_encoders
