from sklearn.metrics import recall_score, classification_report, accuracy_score

def random_forest(model, X_train, y_train, X_test, y_test):
    # Fit model training
    model.fit(X_train, y_train)
    # Get predictions
    pred = model.predict(X_test)
    # Get metrics
    classif_rep = classification_report(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    rec = recall_score(y_test, pred)
    return (f"""Classification report: {classif_rep},
Accuracy: {accuracy},
Recall : {rec}""")

