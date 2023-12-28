from sklearn.neural_network import MLPRegressor
import  joblib

class NeuralNetworkRegressor:
    def __init__(self, hidden_layers=(256, 256, 256), alpha=0.001, max_iter=1000):
        """
        Initialize the MLP Regressor with specified parameters.
        """
        self.hidden_layers = hidden_layers
        self.alpha = alpha
        self.max_iter = max_iter
        self.model = MLPRegressor(
            activation='relu',
            alpha=self.alpha,
            hidden_layer_sizes=self.hidden_layers,
            learning_rate='adaptive',
            solver='adam',
            max_iter=self.max_iter,
            verbose=1,
        )

    def fit(self, X_train, y_train):
        """
        Fit the MLP model to the training data.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Make predictions with the MLP model.
        """
        return self.model.predict(X)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        """
        return {
            'hidden_layers': self.hidden_layers,
            'alpha': self.alpha,
            'max_iter': self.max_iter
        }

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        # Reinitialize the model with new parameters
        self.model = MLPRegressor(
            activation='relu',
            alpha=self.alpha,
            hidden_layer_sizes=self.hidden_layers,
            learning_rate='adaptive',
            solver='adam',
            max_iter=self.max_iter,
            verbose=1,
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
