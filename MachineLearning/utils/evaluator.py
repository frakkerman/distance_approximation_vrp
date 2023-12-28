import numpy as np
from sklearn import metrics
import math

class Evaluator:
    def __init__(self, actual,n_features):
        self.actual = actual
        self.predicted = None
        self.n_features = n_features

    def _error(self):
        """ Simple error """
        return self.actual - self.predicted

    def _percentage_error(self):
        """ Percentage error (result is NOT multiplied by 100) """
        return self._error() / self.actual

    def mpe(self):
        """ Mean Percentage Error """
        return np.mean(self._percentage_error()) * 100

    def mae(self):
        """ Mean Absolute Error """
        return metrics.mean_absolute_error(self.actual, self.predicted)

    def rmse(self):
        """ Root Mean Squared Error """
        return math.sqrt(metrics.mean_squared_error(self.actual, self.predicted))

    def r_squared(self):
        """ R-squared Score """
        return metrics.r2_score(self.actual, self.predicted)

    def adjusted_r_squared(self, n_samples, n_features):
        """ Adjusted R-squared Score """
        r2 = self.r_squared()
        return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

    def mape(self):
        """ Mean Absolute Percentage Error """
        return metrics.mean_absolute_percentage_error(self.actual, self.predicted) * 100

    def print_evaluation_metrics(self, prediction, save_to_file=False, file_path="evaluation_metrics.txt"):
        n_samples = prediction.shape[0]
        self.predicted = prediction

        metrics = [
            f'Mean Absolute Error: {self.mae()}',
            f'RMSE: {self.rmse()}',
            f'rRMSE: {self.rmse() / np.mean(self.actual) * 100}%',
            f'rMAE: {self.mae() / np.mean(self.actual) * 100}%',
            f'R-squared: {self.r_squared()}',
            f'Adjusted R-squared: {self.adjusted_r_squared(n_samples, self.n_features)}',
            f'MAPE: {self.mape()}',
            f'MPE: {self.mpe()}',
            '-----------------------'
        ]

        # Print metrics
        for metric in metrics:
            print(metric)

        # Save metrics to file if requested
        if save_to_file:
            with open(file_path, 'w') as file:
                for metric in metrics:
                    file.write(metric + '\n')

