# Machine Learning

In this project, we load the engineered features and train and evaluate the machine learning models.

In `main.py` you need to set the parameters to the same settings as used for generating the data, such that the same hash is used for loading the data. Next, you can set several parameters related to training, e.g., whether ot not to normalize the data, wheter to do feature selection, whether to use Bayesian optimization to tune the ML-models, or to altenrtaively use a brute force grid search. 

After training, we print performanc statistics for all ML-models.