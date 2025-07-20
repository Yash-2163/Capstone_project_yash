# Loading data from local

import pandas as pd
import os

def load_data(target_col='Converted'):
    """
    Loads raw data and splits into features and target.
    No preprocessing here.
    
    Parameters:
    - target_col (str): Name of the target column in the dataset. Default is 'Converted'.

    Returns:
    - X (DataFrame): Features (all columns except target_col)
    - y (Series): Target variable (converted column)
    """

    # Define the base directory for your project within the Docker container

    # BASE_DIR = "/opt/airflow"

    # Construct the full path to the CSV file containing your sample data
    # This assumes the file is stored at /opt/airflow/data/sample_data2.csv inside the container

    file_path = os.path.join("..", 'data', 'Lead_scoring.csv')

    # Check if the file actually exists, and raise an error if not
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Data file not found at: {file_path}. Please ensure it's in the 'data' directory "
            "under the project root (/opt/airflow/data/)."
        )

    # Read the CSV into a DataFrame
    df = pd.read_csv(file_path)

    # Separate features (X) and target (y)
    X = df.drop(columns=[target_col])  # Drop the target column from features
    y = df[target_col]                 # Isolate the target column

    return X, y
