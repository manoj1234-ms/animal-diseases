from xgboost import XGBClassifier

def get_stage1_model():
    return XGBClassifier(
        objective="multi:softprob",
        num_class=6,
        eval_metric="mlogloss",
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        random_state=42
    )
