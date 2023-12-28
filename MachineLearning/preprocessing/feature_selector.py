import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV

class RecursiveFeatureSelection:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features='auto', random_state=42, 
                 step=1, cv=5, n_jobs=-1):
        """
        Initialize the Recursive Feature Selection class with a Random Forest Regressor.
        :param n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, random_state: 
               Parameters for the Random Forest Regressor.
        :param step: Number of features to remove at each iteration.
        :param cv: Number of folds for cross-validation.
        :param n_jobs: Number of jobs to run in parallel.
        """
        self.estimator = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, 
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
            max_features=max_features, random_state=random_state,n_jobs=n_jobs
        )
        self.selector = RFECV(self.estimator, step=step, cv=cv, n_jobs=n_jobs)
        self.feature_idx = None
        self.grid_scores = None

    def fit(self, X, y):
        """
        Fit the RFECV selector to the data.
        :param X: Training data features.
        :param y: Training data target.
        """
        self.selector.fit(X, y)
        self.feature_idx = self.selector.get_support()
        self.grid_scores = self.selector.grid_scores_
        
    def transform(self, X):
        """
        Apply the feature selection to new data.
        :param X: Data to be transformed.
        :return: Transformed data with selected features.
        """
        if self.feature_idx is None:
            raise ValueError("Fit the model before applying transform.")
        return X[:, self.feature_idx]

    def get_selected_features(self, feature_names):
        """
        Get the names of the selected features.
        :param feature_names: List of all feature names.
        :return: List of selected feature names.
        """
        if self.feature_idx is None:
            raise ValueError("Fit the model before getting selected features.")
        return feature_names[self.feature_idx]

    def plot_feature_selection(self):
        """
        Plot the feature selection process.
        """
        if self.grid_scores is None:
            raise ValueError("Fit the model before plotting feature selection.")
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score of number of selected features")
        plt.plot(range(1, len(self.grid_scores) + 1), self.grid_scores)
        plt.show()
