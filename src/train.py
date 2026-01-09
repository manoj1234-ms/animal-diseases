# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split

# def train_model(df, preprocessor, model):
#     X = df.drop("target_disease", axis=1)
#     y = df["target_disease"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y,
#         test_size=0.2,
#         stratify=y,
#         random_state=42
#     )

#     pipeline = Pipeline([
#         ("preprocessing", preprocessor),
#         ("model", model)
#     ])

#     pipeline.fit(X_train, y_train)

#     return pipeline, X_test, y_test


# src/train.py

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

def train_model(df, preprocessor, model):
    X = df.drop("target_disease", axis=1)
    y = df["target_disease"]

    # Encode target labels (REQUIRED for XGBoost)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test, label_encoder
