from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from typing import Dict, Any
import logging
from datetime import datetime
from mlflow import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Weather Classification API",
    description="API for weather classification using ML model",
    version="1.0.0"
)

# Global model variable
model = None

client = MlflowClient()
mlflow.set_tracking_uri("http://18.208.146.199:5000/") 

# Define WeatherInfo based on your actual data structure
class WeatherInfo(BaseModel):
    temperature_2m: float
    relative_humidity_2m: int
    dew_point_2m: float
    apparent_temperature: float
    pressure_msl: float
    cloudcover: int
    cloudcover_low: int
    cloudcover_mid: int
    cloudcover_high: int
    windspeed_10m: int
    windgusts_10m: int
    winddirection_10m: int
    diffuse_radiation: float
    direct_radiation: float
    timestamp: datetime

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    timestamp: datetime

# load the model from mlflow model registry
def load_model_from_mlflow():
    """Load the latest model from MLflow Model Registry"""
    try:
        model_name = "my_model"
        model_version = 2
        model = mlflow.xgboost.load_model(f"models:/{model_name}/{model_version}")
        logger.info(f"Model loaded from MLflow")
        return model
    except Exception as e:
        logger.error(f"Error loading model from MLflow: {e}")
        raise

# load the model from local file system
def load_local_model():
    """Fallback to load local model"""
    try:
        with open("models/model.pkl", 'rb') as f:
            model = pickle.load(f)
        logger.info("Local model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading local model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """load model on startup"""
    global model
    try:
        # try loading from MLflow first, fallback to local
        try:
            model = load_model_from_mlflow()
        except:
            model = load_local_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# load the scaler object
def load_scaler():
    try:
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded successfully")
        return scaler
    except Exception as e:
        logger.error(f"Error loading scaler: {e}")
        raise

# preprocess the input data that we get from the user
def preprocess_input(weather_data: WeatherInfo) -> np.ndarray:
    # Convert to dictionary
    data = weather_data.dict()
    df = pd.DataFrame([data])
    
    # Extract datetime features
    df["weekday"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute

    # remove timestamp col
    df = df.drop(columns='timestamp')

    # load the scaler
    scaler = load_scaler()

    # scale the data
    X = scaler.transform(df)

    return X

def get_prediction_label(prediction: int) -> str:
    """Convert prediction to human readable label"""
    labels = {
        0: "Clear sky",
        1: "Overcast", 
        2: "Rainy"
    }
    return labels.get(prediction, "Unknown")

@app.get("/")
async def root():
    return {"message": "Weather Classification API is running!"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now()
    }

# Modified to accept WeatherInfo directly
@app.post("/predict", response_model=PredictionResponse)
async def predict(weather_data: WeatherInfo):
    """Make weather prediction - accepts weather data directly"""
    global model
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Preprocess input
        input_data = preprocess_input(weather_data)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        confidence = float(max(prediction_proba))
        
        # Convert to response format
        prediction_label = get_prediction_label(int(prediction))
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_label=prediction_label,
            confidence=confidence,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Alternative endpoint that accepts the wrapped format
@app.post("/predict_wrapped", response_model=PredictionResponse)
async def predict_wrapped(request: dict):
    """Alternative endpoint that accepts wrapped format"""
    global model
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Extract weather_data from request
        weather_data = WeatherInfo(**request["weather_data"])
        
        # Preprocess input
        input_data = preprocess_input(weather_data)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        confidence = float(max(prediction_proba))
        
        # Convert to response format
        prediction_label = get_prediction_label(int(prediction))
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_label=prediction_label,
            confidence=confidence,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(weather_data_list: list[WeatherInfo]):
    """Batch prediction endpoint - accepts list of weather data directly"""
    global model
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        results = []
        for weather_data in weather_data_list:
            input_data = preprocess_input(weather_data)
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            confidence = float(max(prediction_proba))
            
            results.append({
                "prediction": int(prediction),
                "prediction_label": get_prediction_label(int(prediction)),
                "confidence": confidence
            })
        
        return {
            "results": results,
            "timestamp": datetime.now(),
            "total_predictions": len(results)
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)