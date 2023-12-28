import pandas as pd

class DataHandler:
    def __init__(self, raw_data_path=None, processed_data_columns=None,nrows=100):
        self.raw_data_path = raw_data_path
        self.raw_data = None
        self.processed_data = None
        self.processed_data_columns = processed_data_columns
        self.nrows = nrows

    def load_data(self):
        self.raw_data = pd.read_csv(self.raw_data_path, sep=';', 
                                    names=["ClusterID", "LocationID", "Lat", "Lon", "ExpFillLevel", 
                                           "Distance", "ServiceLevel"], index_col=False,nrows=self.nrows)

    def initialize_processed_data(self):
        if self.processed_data_columns:
            self.processed_data = pd.DataFrame(columns=self.processed_data_columns)

    def append_to_processed_data(self, new_data_dict):
        new_data_df = pd.DataFrame([new_data_dict])
        self.processed_data = pd.concat([self.processed_data, new_data_df], ignore_index=True)

    def save_data(self, file_path, data):
        # Save the provided data to the specified file path
        data.to_csv(file_path, index=False)
