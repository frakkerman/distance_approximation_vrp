from xgboost import XGBRegressor
import joblib

class XGBRegressorModel:
    def __init__(self, n_estimators=200, random_state=42):
        """
        Initialize the XGB Regressor model with specified parameters.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )

    def fit(self, X_train, y_train):
        """
        Fit the XGB model to the training data.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Make predictions using the XGB model.
        """
        return self.model.predict(X)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        """
        return {
            'n_estimators': self.n_estimators,
            'random_state': self.random_state
        }

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        # Reinitialize the model with new parameters
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
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
