import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("errors.log")
file_handler.setLevel('ERROR')


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


# load the data
def load_data(file_name : str) -> pd.DataFrame:
    data_dir = os.path.join('data', 'external')
    data_path = os.path.join(data_dir, file_name)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
        logger.error(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Data loaded successfully from {data_path}")
    return df


# split the data into train and test sets
def split_data(file_name: str):
    # load the data
    df = load_data(file_name)
    X = df.drop(columns=['weather_code_binary'])
    y = df['weather_code_binary'].values
    logger.info("Data split into features and target variable.")
    return X, y

# split the data into train and test sets
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        data_path = os.path.join(data_path, 'processed')
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, 'train_scaled.csv'), index=False)
        test_data.to_csv(os.path.join(data_path, 'test_scaled.csv'), index=False)
        logger.debug('data saved')  
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data.")
        print(e)
        raise

# feature engineering function
def feature_engineering() -> pd.DataFrame:
    # scale the data
    scaler = StandardScaler()
    X_train, y_train = split_data('train_clean.csv')
    X_test, y_test = split_data('test_clean.csv')

    _train_scale  = scaler.fit_transform(X_train)
    _test_data = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(_train_scale, columns=X_train.columns)
    X_train_scaled['weather_code_binary'] = y_train

    X_test_scaled = pd.DataFrame(_test_data, columns=X_test.columns)
    X_test_scaled['weather_code_binary'] = y_test
    logger.info("Feature engineering completed successfully.")
    return X_train_scaled, X_test_scaled

def main():
    X_train_scaled, X_test_scaled = feature_engineering()

    save_data(X_train_scaled, X_test_scaled, data_path='data')


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
