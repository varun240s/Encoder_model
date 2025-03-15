from sklearn.metrics import classification_report

def evaluate_model(y_true, y_pred):
    report=classification_report(y_true,y_pred,target_names=["Negative", "Positive"])
    print(report)
    