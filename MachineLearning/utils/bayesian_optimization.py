from numpy import mean
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.utils import use_named_args
import numpy as np

class BayesianOptimization:
    def __init__(self, model, x_train, y_train, search_space, cv=3, scoring='r2', n_calls=20, n_random_starts=5):
        """
        Initialize the Bayesian Optimization class.
        :param model: The machine learning model to be optimized.
        :param x_train: Training data features.
        :param y_train: Training data target.
        :param search_space: List of hyperparameters to search.
        :param cv: Number of folds for cross-validation.
        :param scoring: Scoring metric for evaluation.
        :param n_calls: Number of iterations for optimization.
        :param n_random_starts: Number of random initialization points.
        """
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.search_space = search_space
        self.cv = cv
        self.scoring = scoring
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts

        # Decorator for evaluate_model method
        self.evaluate_model = use_named_args(search_space)(self.evaluate_model)

    def evaluate_model(self, **params):
        """
        Evaluate the model with the given hyperparameters.
        :param params: Hyperparameters to test.
        :return: The negative mean score of the cross-validation.
        """
        self.model.set_params(**params)
        result = cross_val_score(self.model, self.x_train, self.y_train, cv=self.cv, n_jobs=-1, scoring=self.scoring)
        estimate = mean(result)
        return 1.0 - estimate

    def optimize(self):
        """
        Perform Bayesian optimization.
        :return: Optimal parameters and score.
        """
        np.int = int#needed for fixing deprecated bug in skopt, see: https://stackoverflow.com/questions/76321820/how-to-fix-the-numpy-int-attribute-error-when-using-skopt-bayessearchcv-in-sci
        result = gp_minimize(self.evaluate_model, self.search_space, n_calls=self.n_calls, n_random_starts=self.n_random_starts)
        optimal_params = {name: result.x[i] for i, name in enumerate([param.name for param in self.search_space])}
        optimal_score = 1.0 - result.fun

        return optimal_params, optimal_score

    def print_optimal_results(self):
        optimal_params, optimal_score = self.optimize()
        print('Best score: %.3f' % optimal_score)
        print('Best Parameters:', optimal_params)
