from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

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
    dist_to_river_m: float = Field(1000.0, description="Distance to nearest river (m)")

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
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True