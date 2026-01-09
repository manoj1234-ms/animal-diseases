from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def get_baseline_model():
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

def get_final_model():
    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=200,
        max_depth=6
    )
