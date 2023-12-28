from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.base import clone

class GridSearchOptimizer:
    def __init__(self, estimator, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=None):
        """
        Initialize the GridSearchOptimizer.
        :param estimator: The machine learning model/estimator for which the grid search will be performed.
        :param param_grid: Dictionary with parameters names as keys and lists of parameter settings to try as values.
        :param cv: Number of folds for cross-validation.
        :param scoring: Strategy to evaluate the performance of the cross-validated model on the test set.
        :param n_jobs: Number of jobs to run in parallel. -1 means using all processors.
        :param random_state: Random state for reproducibility.
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_estimator_ = None

    def perform_grid_search(self, X, y):
        """
        Perform Grid Search Cross-Validation.
        :param X: Training data features.
        :param y: Training data target.
        :return: Best parameters from grid search.
        """
        gsc = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            verbose=0,
            n_jobs=self.n_jobs
        )
        gsc.fit(X, y)
        self.best_estimator_ = gsc.best_estimator_
        return gsc.best_params_

    def evaluate_best_model(self, X, y, cv=3):
        """
        Evaluate the best model from grid search using K-Fold cross-validation.
        :param X: Training data features.
        :param y: Training data target.
        :param cv: Number of folds for cross-validation.
        :return: Cross-validation scores for the best model.
        """
        if self.best_estimator_ is None:
            raise ValueError("Perform grid search before evaluating the best model.")

        cloned_estimator = clone(self.best_estimator_)
        cloned_estimator.random_state = self.random_state
        scores = cross_val_score(cloned_estimator, X, y, cv=cv, scoring=self.scoring, n_jobs=self.n_jobs)
        return scores
