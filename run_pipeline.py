
# from src.data_loader import load_data
# from src.target_builder import create_target_column
# from src.preprocessing import build_preprocessor
# from src.model import get_final_model
# from src.train import train_model
# from src.evaluate import evaluate_model

# # -----------------------------
# # Configuration
# # -----------------------------
# DATA_PATH = "data/full_animal_disease_dataset.csv"

# # -----------------------------
# # Step 1: Load data
# # -----------------------------
# df = load_data(DATA_PATH)
# print("Initial columns:")
# print(df.columns)

# # -----------------------------
# # Step 2: Create target column
# # -----------------------------
# df = create_target_column(df)
# print("\nColumns after target creation:")
# print(df.columns)

# # -----------------------------
# # Step 3: Build preprocessing pipeline
# # -----------------------------
# preprocessor = build_preprocessor(df)

# # -----------------------------
# # Step 4: Get final model
# # -----------------------------
# model = get_final_model()

# # -----------------------------
# # Step 5: Train model
# # -----------------------------
# trained_model, X_test, y_test, label_encoder = train_model(
#     df=df,
#     preprocessor=preprocessor,
#     model=model
# )

# # -----------------------------
# # Step 6: Evaluate model
# # -----------------------------
# # evaluate_model(trained_model, X_test, y_test)4
# evaluate_model(trained_model, X_test, y_test, label_encoder)


from src.data_loader import load_data
from src.target_builder import create_targets
from src.preprocessing import build_preprocessor
from src.model_stage1 import get_stage1_model
from src.train_stage1 import train_stage1
from src.train_stage2 import train_stage2_models
from src.evaluate import evaluate_model
from src.save_models import save_all_models

DATA_PATH = "data/full_animal_disease_dataset.csv"

df = load_data(DATA_PATH)
df = create_targets(df)

print(df.columns.tolist())


preprocessor = build_preprocessor(df)

# -------- Stage 1 --------
stage1_model = get_stage1_model()

stage1_pipeline, X_test, y_test, stage1_label_encoder = train_stage1(
    df=df,
    preprocessor=preprocessor,
    model=stage1_model
)

evaluate_model(stage1_pipeline, X_test, y_test, stage1_label_encoder)

# -------- Stage 2 --------
stage2_models, disease_label_encoders = train_stage2_models(df, preprocessor)

print(f"Stage-2 models trained: {list(stage2_models.keys())}")

# -------- Save --------
save_all_models(
    stage1_pipeline,
    stage2_models,
    stage1_label_encoder,
    disease_label_encoders
)
