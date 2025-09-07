from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from sklearn.preprocessing import StandardScaler
from models import WeatherInfo
import uvicorn
import pandas as pd
import numpy as np
import pickle

app = FastAPI()

# load the scaler object
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# load the model
with open("models/model.pkl", 'rb') as f:
    model = pickle.load(f)


@app.get("/")
def home():
    return {"Hello, World"}


@app.post("/predict")
def predict(weatherInfo: WeatherInfo):
    data = dict(weatherInfo)
    df = pd.DataFrame([data])
    df["weekday"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute

    # remove timestamp col
    df = df.drop(columns='timestamp')

    X = scaler.transform(df)
    
    prediction = model.predict(X)

    if prediction[0] == 0:
        return {"prediction": "Clear Sky"}
    elif prediction[0] == 1:
        return {"prediction": "Overcast"}
    else:
        return {"prediction": "Rain"}
    # return prediction

if __name__ == "__main__":
    uvicorn.run(app="main:app", port=8501, reload=True, host="0.0.0.0")


