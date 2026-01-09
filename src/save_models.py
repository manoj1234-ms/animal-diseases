import joblib
import os

def save_all_models(
    stage1_pipeline,
    stage2_models,
    category_label_encoder,
    disease_label_encoders
):
    os.makedirs("models", exist_ok=True)

    joblib.dump(stage1_pipeline, "models/stage1_pipeline.pkl")
    joblib.dump(stage2_models, "models/stage2_models.pkl")
    joblib.dump(category_label_encoder, "models/category_encoder.pkl")
    joblib.dump(disease_label_encoders, "models/disease_encoders.pkl")

    print("Models saved successfully")
