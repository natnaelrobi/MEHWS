from datetime import datetime
from sqlalchemy import Column, DateTime, Float, Integer, String
from app.database import Base

class WeatherTelemetry(Base):
    __tablename__ = "weather_telemetry"

    id = Column(Integer, primary_key=True, index=True)
    region_key = Column(String, index=True)
    pcode = Column(String, index=True, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Telemetry metrics
    rain_24h = Column(Float, default=0.0)
    rain_72h = Column(Float, default=0.0)
    rain_30d = Column(Float, default=0.0)          # Core Drought Metric (Cumulative)
    soil_moisture_mean = Column(Float, default=0.0) # Core Drought/Flood Metric
    ndvi_mean = Column(Float, nullable=True)        # Vegetation Vigor (Drought)
    
    # Computed baseline hazard risks (%)
    flash_flood_risk = Column(Float, default=0.0)
    drought_risk = Column(Float, default=0.0)
    landslide_risk = Column(Float, nullable=True)  # Optional/Secondary Hazard