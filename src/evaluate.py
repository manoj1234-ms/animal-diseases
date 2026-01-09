# from sklearn.metrics import classification_report

# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     report = classification_report(y_test, y_pred)
#     print(report)


# src/evaluate.py

from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)

    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    print("\nClassification Report:")
    print(classification_report(y_test_decoded, y_pred_decoded))
