# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer

# def build_preprocessor(df):
#     X = df.drop("target_disease", axis=1)
    
#     categorical_cols = ["Animal", "Gender", "Breed"]

#     numerical_cols = [
#     col for col in X.columns
#     if col not in categorical_cols
# ]

    
#     num_pipeline = Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler())
#     ])
    
#     cat_pipeline = Pipeline([
#         ("imputer", SimpleImputer(strategy = "most_frequent")),
#         ("encoder", OneHotEncoder(handle_unknown="ignore"))
#     ])
    
#     preprocessor = ColumnTransformer([
#         ("num", num_pipeline, numerical_cols)
#         ("cat", cat_pipeline, categorical_cols)
#     ])
    
#     return preprocessor

# src/preprocessing.py
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def build_preprocessor(df):
    # 👉 CHANGED HERE (was: df.drop("target_disease", axis=1))
    X = df.drop(
        columns=[c for c in ["target_disease", "target_category"] if c in df.columns]
    )

    categorical_cols = ["Animal", "Gender", "Breed"]

    numerical_cols = [
        col for col in X.columns
        if col not in categorical_cols
    ]

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ]
    )

    return preprocessor
