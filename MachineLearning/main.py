from preprocessing.data_loader import DataLoader
from preprocessing.normalizer import Normalizer
from preprocessing.feature_selector import RecursiveFeatureSelection
from models.neural_networks import NeuralNetworkRegressor
from models.random_forrests import RandomForestRegressorModel
from models.linear_regression import LinearRegressionModel
from models.lightgbm import LGBMRegressorModel
from models.xgboost import XGBRegressorModel
from utils.bayesian_optimization import BayesianOptimization
from utils.evaluator import Evaluator
from utils.grid_search import GridSearchOptimizer
from skopt.space import Real, Integer
import numpy as np

def main():
    # User settings for gettig the right file
    params = {
        'polygon_nr1': 10,
        'polygon_nr2': 15,
        'circle_size_1': 0.5,
        'circle_size_2': 0.75,
        'vehicle_capacity': 8000
    }
    raw_data_file_name = "example_vrp_data_AMS"  # file name of raw data
    base_dir = "../data"
    
    # Initialize and load data
    data_loader = DataLoader(base_dir, params, raw_data_file_name)
    data_loader.load_data()
    data_loader.split_data()
    x_train, y_train = data_loader.get_train_data()
    x_test, y_test = data_loader.get_test_data()

    # User settings
    settings = {
        "normalize": True,
        "use_feature_selection": False,#using random forests recursive feature elemination
        "use_bayesian_optimization": False,#for hyperparam tuning
        "use_grid_search": False,#alternative hyperparam tuning method
        # Additional settings as required
    }

    # Normalization or Standardization
    normalizer = Normalizer(method='normalize' if settings["normalize"] else None)#alternatively, you can standardize here
    x_train = normalizer.fit_transform(x_train)
    x_test = normalizer.transform(x_test)

    # Initialize models
    models = {
        "Linear Regression": LinearRegressionModel(),
        "Random Forest": RandomForestRegressorModel(),
        "LightGBM": LGBMRegressorModel(),
        "XGBoost": XGBRegressorModel(),
        "Neural Network": NeuralNetworkRegressor()
    }

    # Feature Selection
    if settings["use_feature_selection"]:
        # Initialize feature selector
        feature_selector = RecursiveFeatureSelection()
    
        # Fit the selector and transform the training data
        feature_selector.fit(x_train, y_train)
        x_train = feature_selector.transform(x_train)
        x_test = feature_selector.transform(x_test)
    
        # Optionally, get selected feature names and plot the feature selection process
        selected_features = feature_selector.get_selected_features(np.array(x_train.columns))
        print("Selected features:", selected_features)
        feature_selector.plot_feature_selection()


    # Train, evaluate, and potentially tune models
    evaluator_train = Evaluator(y_train,x_train.shape[1])
    evaluator_test = Evaluator(y_test,x_test.shape[1])
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        model.save_model("models/saved_models/"+name)
        print(f"Evaluating {name} on training data...")
        evaluator_train.print_evaluation_metrics(model.predict(x_train))
        print(f"Evaluating {name} on training data...")
        evaluator_test.print_evaluation_metrics(model.predict(x_test))

        # Hyperparameter Tuning
        if settings["use_bayesian_optimization"]:
            print(f"Bayesian optimization for {name} on training data...")
            # Example Bayesian Optimization parameters (to be customized)
            search_space = [
               Real(0.01, 1.0, name='learning_rate'),
               Integer(100, 1000, name='n_estimators')
           ]
            bayesian_optimizer = BayesianOptimization(model, x_train, y_train, search_space)
            bayesian_optimizer.optimize()
        if settings["use_grid_search"]:
            print(f"Grid search for {name} on training data...")
            # Example Grid Search parameters (to be customized)
            param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
            grid_search = GridSearchOptimizer(model, param_grid)
            best_params = grid_search.perform_grid_search(x_train, y_train)
            print(f"Best parameters for {name}:", best_params)

if __name__ == "__main__":
    main()
