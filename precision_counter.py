from sklearn.metrics import precision_score


def precision_func(model, x_train, y_train, x_test, y_test):
    # Fit model training
    model.fit(x_train, y_train)
    # Get predictions
    prediction = model.predict(x_test)
    # Get metrics
    precision = precision_score(y_test, prediction)
    return precision
