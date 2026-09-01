from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional

from app.database import get_db, HazardPrediction
from app.models.telemetry import WeatherTelemetry
from app.models.prediction import generate_region_prediction
from app.services.open_meteo import REGIONS_COORDS, fetch_weather_forecast

router = APIRouter(prefix="/predictions", tags=["Hazard Predictions"])

# 1. Define Pydantic Schemas
class HazardPredictionRequest(BaseModel):
    latitude: float = Field(..., example=9.03)
    longitude: float = Field(..., example=38.74)
    woreda_name: Optional[str] = Field(None, example="Bole")
    rfh: float = Field(0.0, description="Short-term rainfall (mm)")
    r1h: float = Field(0.0, description="1-Month cumulative rainfall (mm)")
    r3h: float = Field(0.0, description="3-Month cumulative rainfall (mm)")
    rfq: float = Field(50.0, description="Quarterly rainfall deficit factor")
    r3q: float = Field(50.0, description="3-Quarter rainfall deficit factor")
    slope_mean: float = Field(5.0, description="Mean terrain slope (degrees)")
    soil_moisture_mean: float = Field(400.0, description="Soil moisture index")
    ndvi_mean: float = Field(0.4, description="NDVI vegetation index")
    dist_to_river_m: float = Field(1000.0, description="Distance to river (m)")

class HazardPredictionResponse(BaseModel):
    id: Optional[int] = None
    latitude: float
    longitude: float
    woreda_name: Optional[str]
    predicted_flood_risk: float
    predicted_drought_risk: float
    predicted_landslide_risk: float
    primary_hazard_type: str
    overall_risk_level: str

    class Config:
        from_attributes = True

# 2. Route Implementation
@router.post("/realtime", response_model=HazardPredictionResponse, status_code=status.HTTP_201_CREATED)
def predict_hazard(payload: HazardPredictionRequest, db: Session = Depends(get_db)):
    features = payload.model_dump()
    prediction_results = generate_region_prediction(features)

    # Save prediction record to database
    db_record = HazardPrediction(
        latitude=payload.latitude,
        longitude=payload.longitude,
        woreda_name=payload.woreda_name,
        predicted_flood_risk=prediction_results["predicted_flood_risk"],
        predicted_drought_risk=prediction_results["predicted_drought_risk"],
        predicted_landslide_risk=prediction_results["predicted_landslide_risk"],
        primary_hazard_type=prediction_results["primary_hazard_type"],
        overall_risk_level=prediction_results["overall_risk_level"]
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    return db_record