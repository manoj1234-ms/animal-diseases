from src import inference
import pandas as pd

# Reuse loaded objects from src.inference
stage1_pipeline = inference.stage1_pipeline
stage2_models = inference.stage2_models
category_encoder = inference.category_encoder
disease_encoders = inference.disease_encoders


def predict_with_confidence(input_dict):
    input_df = pd.DataFrame([input_dict])

    # Stage 1
    pre1 = stage1_pipeline.named_steps.get('preprocessor')
    model1 = stage1_pipeline.named_steps.get('model')
    Xt1 = pre1.transform(input_df)
    n1 = getattr(model1, 'n_features_in_', None)
    if n1 is not None and Xt1.shape[1] != n1:
        if Xt1.shape[1] < n1:
            import numpy as np
            pad = np.zeros((Xt1.shape[0], n1 - Xt1.shape[1]))
            Xt1 = np.hstack([Xt1, pad])
        else:
            Xt1 = Xt1[:, :n1]

    cat_encoded = model1.predict(Xt1)[0]
    cat = category_encoder.inverse_transform([cat_encoded])[0]
    stage1_conf = None
    if hasattr(model1, 'predict_proba'):
        try:
            stage1_conf = float(model1.predict_proba(Xt1).max())
        except Exception:
            stage1_conf = None

    # Stage 2
    stage2_pipeline = stage2_models.get(cat)
    if stage2_pipeline is None:
        return {
            'predicted_category': cat,
            'predicted_disease': 'Unknown',
            'stage1_confidence': stage1_conf,
            'stage2_confidence': None
        }

    pre2 = stage2_pipeline.named_steps.get('preprocessor')
    model2 = stage2_pipeline.named_steps.get('model')
    Xt2 = pre2.transform(input_df)
    n2 = getattr(model2, 'n_features_in_', None)
    if n2 is not None and Xt2.shape[1] != n2:
        import numpy as np
        if Xt2.shape[1] < n2:
            pad = np.zeros((Xt2.shape[0], n2 - Xt2.shape[1]))
            Xt2 = np.hstack([Xt2, pad])
        else:
            Xt2 = Xt2[:, :n2]

    disease_encoded = model2.predict(Xt2)[0]
    disease = disease_encoders[cat].inverse_transform([disease_encoded])[0]
    stage2_conf = None
    if hasattr(model2, 'predict_proba'):
        try:
            stage2_conf = float(model2.predict_proba(Xt2).max())
        except Exception:
            stage2_conf = None

    return {
        'predicted_category': cat,
        'predicted_disease': disease,
        'stage1_confidence': round(stage1_conf, 3) if stage1_conf is not None else None,
        'stage2_confidence': round(stage2_conf, 3) if stage2_conf is not None else None
    }


sample_inputs = [
    # similar to previous
    {
        'Animal': 'Dog', 'Age': 4, 'Gender': 'Male', 'Breed': 'Labrador',
        'WBC': 12000, 'RBC': 5.1, 'Hemoglobin': 13.5, 'Platelets': 250000,
        'Glucose': 95, 'ALT': 45, 'AST': 40, 'Urea': 30, 'Creatinine': 1.1,
        'Symptom_Fever': 1, 'Symptom_Lethargy': 1, 'Symptom_Vomiting': 0,
        'Symptom_Diarrhea': 1, 'Symptom_WeightLoss': 0, 'Symptom_SkinLesion': 0
    },
    # more febrile
    {
        'Animal': 'Cat', 'Age': 2, 'Gender': 'Female', 'Breed': 'Breed1',
        'WBC': 18000, 'RBC': 4.5, 'Hemoglobin': 11.0, 'Platelets': 180000,
        'Glucose': 110, 'ALT': 120, 'AST': 90, 'Urea': 25, 'Creatinine': 0.8,
        'Symptom_Fever': 1, 'Symptom_Lethargy': 1, 'Symptom_Vomiting': 1,
        'Symptom_Diarrhea': 0, 'Symptom_WeightLoss': 1, 'Symptom_SkinLesion': 0
    },
    # parasitic-like
    {
        'Animal': 'Turtle', 'Age': 5, 'Gender': 'Male', 'Breed': 'Breed2',
        'WBC': 8000, 'RBC': 3.0, 'Hemoglobin': 9.5, 'Platelets': 120000,
        'Glucose': 60, 'ALT': 30, 'AST': 20, 'Urea': 10, 'Creatinine': 0.3,
        'Symptom_Fever': 0, 'Symptom_Lethargy': 1, 'Symptom_Vomiting': 0,
        'Symptom_Diarrhea': 0, 'Symptom_WeightLoss': 1, 'Symptom_SkinLesion': 1
    },
    # metabolic-like
    {
        'Animal': 'Snake', 'Age': 3, 'Gender': 'Female', 'Breed': 'Breed3',
        'WBC': 7000, 'RBC': 2.8, 'Hemoglobin': 8.5, 'Platelets': 150000,
        'Glucose': 300, 'ALT': 40, 'AST': 35, 'Urea': 80, 'Creatinine': 2.0,
        'Symptom_Fever': 0, 'Symptom_Lethargy': 0, 'Symptom_Vomiting': 0,
        'Symptom_Diarrhea': 0, 'Symptom_WeightLoss': 1, 'Symptom_SkinLesion': 0
    }
]

if __name__ == '__main__':
    for i, inp in enumerate(sample_inputs, 1):
        out = predict_with_confidence(inp)
        print(f"Sample {i}:", out)
