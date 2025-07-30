import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV
import os
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
import os



# load the data
def load_data(path : str) -> pd.DataFrame:
    current_dir = os.path.join(os.getcwd())
    parent_dir = os.path.dirname(current_dir)
    data_path = os.path.join(parent_dir, path)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    return df

    # data loading
df = load_data('data/raw/weather_data.csv')


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

target = df['weather_code'].map(weather_code_map).values

main_df = df[features]

main_df.insert(1, 'weather_code_map', target)


# precipitation drop
main_df.drop('precipitation', axis=1, inplace=True)

main_df['weather_code_binary'] = main_df['weather_code_map'].map({
    'Clear sky': 0,
    'Overcast': 1,
    'Slight rain': 2,
    'Moderate rain': 2,
    'Heavy rain': 2

})


df_binary = main_df[main_df['weather_code_map'].isin(['Clear sky', 'Overcast', 'Slight rain', 'Moderate rain', 'Heavy rain'])].copy()
df_binary['weather_code_binary'] = df_binary['weather_code_map'].map({
    'Clear sky': 0,
    'Overcast': 1,
    'Slight rain': 2,
    'Moderate rain': 2,
    'Heavy rain': 2

})

# rem terrestrial_radiation, sunshine_duration, shortwave_radiation
df_binary.drop(columns=['terrestrial_radiation', 'sunshine_duration', 'shortwave_radiation'], inplace=True)

# remove the 'weather_code_map' column
df_binary.drop(columns=['weather_code_map'], inplace=True)

# convert time to datetime
df_binary['time'] = pd.to_datetime(df_binary['time'])

# extract date, time, month, year, hour, minute
df_binary['weekday'] = df_binary['time'].dt.weekday
df_binary['times'] = df_binary['time'].dt.time
df_binary['month'] = df_binary['time'].dt.month
df_binary['year'] = df_binary['time'].dt.year
df_binary['hour'] = df_binary['time'].dt.hour
df_binary['minute'] = df_binary['time'].dt.minute

df_binary.drop('time', axis=1, inplace=True)

df_binary.drop('times', axis=1, inplace=True)

X = df_binary.drop('weather_code_binary', axis=1)
y = df_binary['weather_code_binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=3, stratify=y)

scale = StandardScaler()
X_train_scale = scale.fit_transform(X_train)
X_test_scale = scale.transform(X_test)



mlflow.set_tracking_uri("http://34.227.105.107:5000/")
mlflow.set_experiment("Xgboost Hyperparameter Tuning")


# params