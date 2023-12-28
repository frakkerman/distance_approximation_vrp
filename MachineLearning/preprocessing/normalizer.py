from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

class Normalizer:
    def __init__(self, method=None):
        """
        Initialize the Normalizer.
        :param method: 'standardize', 'normalize', or None. Determines the method to be applied.
        """
        self.method = method
        self.scaler = None
        if method == 'standardize':
            self.scaler = StandardScaler()
        elif method == 'normalize':
            self.scaler = MinMaxScaler()
        elif method is not None:
            raise ValueError("Method should be 'standardize', 'normalize', or None.")

    def fit_transform(self, data):
        """
        Fit to data and transform it.
        :param data: DataFrame or array-like, the data to be transformed.
        :return: Transformed data.
        """
        if self.scaler is not None:
            return pd.DataFrame(self.scaler.fit_transform(data), columns=data.columns, index=data.index)
        return data

    def transform(self, data):
        """
        Transform the data based on the fitted scaler.
        :param data: DataFrame or array-like, the data to be transformed.
        :return: Transformed data.
        """
        if self.scaler is not None:
            return pd.DataFrame(self.scaler.transform(data), columns=data.columns, index=data.index)
        return data