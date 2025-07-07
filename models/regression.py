
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def fit_linear_model(X, y):
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    return model

def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    return {
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'y_pred': y_pred
    }
