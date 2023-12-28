from sklearn.linear_model import LinearRegression
import joblib

class LinearRegressionModel:
    def __init__(self, fit_intercept=True):
        """
        Initialize the Linear Regression model.
        :param fit_intercept: Whether to calculate the intercept for this model.
        """
        self.fit_intercept = fit_intercept
        self.model = LinearRegression(fit_intercept=self.fit_intercept)

    def fit(self, X_train, y_train):
        """
        Fit the Linear Regression model to the training data.
        :param X_train: Training data features.
        :param y_train: Training data target.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Make predictions using the Linear Regression model.
        :param X: Data features for which to make predictions.
        :return: Predicted values.
        """
        return self.model.predict(X)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        """
        return {'fit_intercept': self.fit_intercept}

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        # Reinitialize the model with new parameters
        self.model = LinearRegression(fit_intercept=self.fit_intercept)
        return self

    def score(self, X, y):
        """
        Returns the coefficient of determination R^2 of the prediction.
        """
        return self.model.score(X, y)
    
    def save_model(self, file_path):
        """
        Save the model to a file.
        :param file_path: Path to the file where the model will be saved.
        """
        joblib.dump(self.model, file_path)

    def load_model(self, file_path):
        """
        Load a model from a file.
        :param file_path: Path to the file from which the model will be loaded.
        """
        self.model = joblib.load(file_path)
