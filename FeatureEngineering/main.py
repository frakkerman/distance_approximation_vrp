import os
import hashlib
from DataHandler import DataHandler
from FeatureEngineer import FeatureEngineer

def generate_unique_id(params, raw_data_file_name):
    # Use only the base file name (without the directory path) and parameters
    unique_str = f"{raw_data_file_name}_param_{params['polygon_nr1']}_{params['polygon_nr2']}_{params['circle_size_1']}_{params['circle_size_2']}_{params['vehicle_capacity']}"
    unique_hash = hashlib.sha256(unique_str.encode()).hexdigest()
    return unique_hash

def main(file_path, output_path):
    # Extract the name of the raw data file
    raw_data_file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Initialize data handler and load raw data
    nrows = 100000  # for testing purpose we limit the number of rows in the data to process
    data_handler = DataHandler(raw_data_path=file_path, nrows=nrows)
    data_handler.load_data()

    depot_loc = (52.2947628, 4.94576789)  # Lat, Lon
    polygon_nr1 = 10
    polygon_nr2 = 15
    circle_size_1 = 0.5
    circle_size_2 = 0.75
    vehicle_capacity = 8000  # used in some feature calc.

    params = {
        'polygon_nr1': polygon_nr1,
        'polygon_nr2': polygon_nr2,
        'circle_size_1': circle_size_1,
        'circle_size_2': circle_size_2,
        'vehicle_capacity': vehicle_capacity
    }

    # Generate a unique identifier hash based on the parameters and raw data file name
    unique_id_hash = generate_unique_id(params, raw_data_file_name)

    # Construct the directory path for saving the data with the unique identifier hash
    unique_output_dir = os.path.join(output_path, "engineered_features")

    # Create the directory if it doesn't exist
    os.makedirs(unique_output_dir, exist_ok=True)

    # Construct the file path for saving the data with the hash as part of the filename
    unique_output_path = os.path.join(unique_output_dir, f"{unique_id_hash}_features_target.csv")

    # Initialize feature engineer with the raw data
    feature_engineer = FeatureEngineer(data_handler.raw_data, depot_loc, polygon_nr1, polygon_nr2, circle_size_1,
                                       circle_size_2, vehicle_capacity)

    # Process the data and calculate features
    feature_engineer.process_data()

    # Get the newly created data
    new_data = feature_engineer.get_new_data()

    # Save the newly created data with the hash as part of the filename
    data_handler.save_data(unique_output_path, new_data)

if __name__ == "__main__":
    input_file_path = '../data/raw_data/example_vrp_data_AMS.txt'  
    output_dir = '../data'  
    main(input_file_path, output_dir)