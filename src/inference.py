# def predict_disease(
#     sample_df,
#     stage1_pipeline,
#     stage2_models,
#     category_label_encoder,
#     disease_label_encoders
# ):
#     """
#     Predict disease for a single input sample
#     """

#     # 1️⃣ Predict disease category
#     category_encoded = stage1_pipeline.predict(sample_df)[0]
#     category = category_label_encoder.inverse_transform(
#         [category_encoded]
#     )[0]

#     # 2️⃣ Select correct stage-2 model
#     stage2_pipeline = stage2_models.get(category)

#     if stage2_pipeline is None:
#         return {
#             "predicted_category": category,
#             "predicted_disease": "Unknown",
#             "confidence": 0.0
#         }

#     # 3️⃣ Predict disease
#     disease_encoded = stage2_pipeline.predict(sample_df)[0]
#     disease = disease_label_encoders[category].inverse_transform(
#         [disease_encoded]
#     )[0]

#     # 4️⃣ Confidence score
#     confidence = stage2_pipeline.predict_proba(sample_df).max()

#     return {
#         "predicted_category": category,
#         "predicted_disease": disease,
#         "confidence": round(float(confidence), 3)
#     }


import pandas as pd
import joblib
import numpy as np

# Load trained models
stage1_pipeline = joblib.load("models/stage1_pipeline.pkl")
stage2_models = joblib.load("models/stage2_models.pkl")
category_encoder = joblib.load("models/category_encoder.pkl")
disease_encoders = joblib.load("models/disease_encoders.pkl")


def predict_disease(input_dict):
    """
    input_dict: dictionary of raw input features
    """

    # 🔹 Convert input to DataFrame (IMPORTANT)
    input_df = pd.DataFrame([input_dict])

    # 🔹 Stage 1: Predict category
    # Transform input using the pipeline's preprocessor and align feature vector
    pre1 = stage1_pipeline.named_steps.get("preprocessor")
    model1 = stage1_pipeline.named_steps.get("model")
    Xt1 = pre1.transform(input_df)
    n1 = getattr(model1, "n_features_in_", None)
    if n1 is not None and Xt1.shape[1] != n1:
        if Xt1.shape[1] < n1:
            pad = np.zeros((Xt1.shape[0], n1 - Xt1.shape[1]))
            Xt1 = np.hstack([Xt1, pad])
        else:
            Xt1 = Xt1[:, :n1]

    category_encoded = model1.predict(Xt1)[0]
    category = category_encoder.inverse_transform([category_encoded])[0]
    # stage1 confidence when available
    stage1_conf = None
    if hasattr(model1, 'predict_proba'):
        try:
            stage1_conf = float(model1.predict_proba(Xt1).max())
        except Exception:
            stage1_conf = None

    # 🔹 Stage 2: Predict disease using category-specific model
    stage2_pipeline = stage2_models[category]
    pre2 = stage2_pipeline.named_steps.get("preprocessor")
    model2 = stage2_pipeline.named_steps.get("model")
    Xt2 = pre2.transform(input_df)
    n2 = getattr(model2, "n_features_in_", None)
    if n2 is not None and Xt2.shape[1] != n2:
        if Xt2.shape[1] < n2:
            pad = np.zeros((Xt2.shape[0], n2 - Xt2.shape[1]))
            Xt2 = np.hstack([Xt2, pad])
        else:
            Xt2 = Xt2[:, :n2]

    disease_encoded = model2.predict(Xt2)[0]
    disease = disease_encoders[category].inverse_transform([disease_encoded])[0]
    stage2_conf = None
    if hasattr(model2, 'predict_proba'):
        try:
            stage2_conf = float(model2.predict_proba(Xt2).max())
        except Exception:
            stage2_conf = None

    return {
        "predicted_category": category,
        "predicted_disease": disease,
        "stage1_confidence": round(stage1_conf, 3) if stage1_conf is not None else None,
        "stage2_confidence": round(stage2_conf, 3) if stage2_conf is not None else None
    }
