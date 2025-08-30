import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging 
import yaml


logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler("errors.log")
file_handler.setLevel('ERROR')


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


# weather description count
weather_code_map = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Slight or moderate thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}


# select relevant features for temprature classification
features = [
    "time",
    'temperature_2m',
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation",
    "pressure_msl",
    "cloudcover",
    "cloudcover_low",
    "cloudcover_mid",
    "cloudcover_high",
    "windspeed_10m",
    "windgusts_10m",
    "winddirection_10m",
    "sunshine_duration",
    "shortwave_radiation",
    "diffuse_radiation",
    "direct_radiation",
    "terrestrial_radiation",
]

def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        test_size = params['data_ingestion']['test_size']
        logger.debug('test size retrieved')
        return test_size
    except FileNotFoundError:
        logger.error('File not found')
        raise
    except yaml.YAMLError as e:
        logger.error('yaml error')
        raise
    except Exception as e:
        logger.error('some error occured')
        raise


# load the data
def load_data(file_name : str) -> pd.DataFrame:
    data_dir = "data/raw/"
    data_path = os.path.join(data_dir, file_name)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
        logger.error(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Data loaded successfully from {data_path}")
    return df


def simple_cleaning(file_name: str) -> pd.DataFrame:
    """Simple preprocessing function to clean the data."""

    # load the data 
    df = load_data(file_name)

    # drop specified columns which not usefull to make prediction
    # precipitation drop
    target = df['weather_code'].map(weather_code_map).values

    df = df[features]

    df.insert(1, 'weather_code_map', target)

    df.drop(columns= ['precipitation','terrestrial_radiation', 'sunshine_duration', 'shortwave_radiation'], inplace=True)

    tem_df = df[df['weather_code_map'].isin(['Clear sky', 'Overcast', 'Slight rain', 'Moderate rain', 'Heavy rain'])]

    
    # select 5 weather codes for binary classification
    tem_df['weather_code_binary'] = tem_df['weather_code_map'].map({
    'Clear sky': 0,
    'Overcast': 1,
    'Slight rain': 2,
    'Moderate rain': 2,
    'Heavy rain': 2

    })

    # drop the weather_code_map column
    tem_df.drop(columns=['weather_code_map'], inplace=True)
    logger.info("Data preprocessing completed successfully.")

    return tem_df

# split the data into train and test sets
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        data_path = os.path.join(data_path, 'interim')
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logger.debug('data saved')
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data.")
        print(e)
        raise


def main():
    try:
        test_size = load_params(params_path='params.yaml')
        clean_df = simple_cleaning('weather_data.csv')
        train_data, test_data = train_test_split(clean_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='data')
        logger.info("Data ingestion completed successfully.")
    except Exception as e:
        print(f"Error: {e}")
        print("Failed to complete the data ingestion process.")


if __name__ == "__main__":
    main()