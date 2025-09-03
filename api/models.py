from pydantic import BaseModel
from datetime import datetime

class WeatherInfo(BaseModel):
    temperature_2m: float
    relative_humidity_2m: float
    dew_point_2m: float
    apparent_temperature: float
    pressure_msl: float
    cloudcover: float
    cloudcover_low: float
    cloudcover_mid: float
    cloudcover_high: float
    windspeed_10m: float
    windgusts_10m: float
    winddirection_10m: float
    diffuse_radiation: float
    direct_radiation: float
    timestamp: datetime