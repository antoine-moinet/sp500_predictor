from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

class LinRegModel(LinearRegression):
    def __init__(self, fit_intercept=True):
        super().__init__(fit_intercept=fit_intercept)

    def evaluate_model(self, X, y):
        y_pred = self.predict(X)
        print("R2:", r2_score(y, y_pred))
        print("MSE:", mean_squared_error(y, y_pred))
