import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
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



# Start parent MLflow run
with mlflow.start_run(run_name="Optuna XGBoost Tuning"):

    def objective(trial):
        with mlflow.start_run(nested=True):

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 0.5),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
                "objective": "multi:softmax",
                "num_class": 3,
                "eval_metric": "mlogloss",
                "tree_method": "hist",
                "use_label_encoder": False,
                "n_jobs": -1,
                "random_state": 42,
            }

            model = XGBClassifier(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            # Log params and metrics
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            return f1

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print("Best trial:")
    print(study.best_trial)

    # Log best parameters and retrain best model
    best_model = XGBClassifier(**study.best_trial.params, 
                               objective="multi:softmax", 
                               num_class=3, 
                               tree_method="hist", 
                               use_label_encoder=False, 
                               random_state=42)

    best_model.fit(X_train, y_train)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "model")
        mlflow.sklearn.save_model(best_model, model_path)
        mlflow.log_artifacts(model_path, "best_model")

    mlflow.log_params(study.best_trial.params)
    mlflow.log_metric("best_f1_score", study.best_value)
