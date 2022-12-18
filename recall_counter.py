from sklearn.metrics import recall_score


def recall_func(model, x_train, y_train, x_test, y_test):
    # Fit model training
    model.fit(x_train, y_train)
    # Get predictions
    prediction = model.predict(x_test)
    # Get metrics
    rec = recall_score(y_test, prediction)
    return rec
