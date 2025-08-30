import pandas as pd
import logging
import os


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
    data_dir = os.path.join('data', 'interim')
    data_path = os.path.join(data_dir, file_name)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
        logger.error(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Data loaded successfully from {data_path}")
    return df

# split the data into train and test sets
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        data_path = os.path.join(data_path, 'external')
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train_clean.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test_clean.csv"), index=False)
        logger.debug('data saved')  
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data.")
        print(e)
        raise

def datetime_extraction(df: pd.DataFrame) -> pd.DataFrame:
    # extract date, time, month, year, hour, minute

    # convert time column to datetime
    df['time']= pd.to_datetime(df['time'])
    
    df['weekday'] = df['time'].dt.weekday
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute

    # drop the time column
    df.drop(columns=['time'], inplace=True)

    logger.info("Datetime extraction completed successfully.")
    return df

def preprocess_data() -> None:
    # load the data
    train_data = load_data("train.csv")
    test_data = load_data("test.csv")

    # extract datetime features
    train_data = datetime_extraction(train_data)
    test_data = datetime_extraction(test_data)

    # save data into external folder
    save_data(train_data, test_data, data_path='data')

    logger.info("Data preprocessing completed successfully.")


if __name__ == "__main__":
    try:
        preprocess_data()
    except Exception as e:
        logger.error(f"An error occurred during data preprocessing: {e}")
        raise


