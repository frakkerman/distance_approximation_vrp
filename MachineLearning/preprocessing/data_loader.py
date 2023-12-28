import pandas as pd
import pandas.errors
from sklearn.model_selection import train_test_split
import os
import hashlib

class DataLoader:
    def __init__(self, base_dir, params, raw_data_file_name, test_size=0.05, random_state=42):
        self.base_dir = base_dir
        self.params = params
        self.raw_data_file_name = raw_data_file_name
        self.test_size = test_size
        self.random_state = random_state
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    @staticmethod
    def generate_unique_id(params, raw_data_file_name):
        unique_str = f"{raw_data_file_name}_param_{params['polygon_nr1']}_{params['polygon_nr2']}_{params['circle_size_1']}_{params['circle_size_2']}_{params['vehicle_capacity']}"
        unique_hash = hashlib.sha256(unique_str.encode()).hexdigest()
        return unique_hash

    def construct_file_path(self):
        unique_id_hash = DataLoader.generate_unique_id(self.params, self.raw_data_file_name)
        load_dir_path = os.path.join(self.base_dir, "engineered_features")
        load_file_path = os.path.join(load_dir_path, f"{unique_id_hash}_features_target.csv")
        return load_file_path

    def load_data(self):
        try:
            file_path = self.construct_file_path()
            self.feature_data = pd.read_csv(file_path)
        except FileNotFoundError:
            print("Error: The specified file was not found. Please check the file path.")
        except pandas.errors.ParserError:
            print("Error: There was an issue parsing the file. Please check if the file format is correct.")
        except Exception as e:
            print("An unexpected error occurred. Please check your file and settings.")
            print(f"Details: {e}")

    def split_data(self):
        if self.feature_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        target = self.feature_data["Target"]
        features = self.feature_data.drop(["Target"], axis=1)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            features, target, test_size=self.test_size, random_state=self.random_state
        )

    def get_train_data(self):
        return self.x_train, self.y_train

    def get_test_data(self):
        return self.x_test, self.y_test
