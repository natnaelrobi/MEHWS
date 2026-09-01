from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class TelemetrySummary(BaseModel):
    rain_24h_mm: float
    rain_72h_mm: float
    rain_30d_mm: Optional[float] = 0.0
    soil_moisture_mean: Optional[float] = None


class HazardScores(BaseModel):
    flash_flood_risk_pct: float
    drought_risk_pct: float
    landslide_risk_pct: Optional[float] = None


class IngestionResponse(BaseModel):
    status: str
    region: str
    primary_hazard_type: Optional[str] = "Flood"
    overall_risk_level: Optional[str] = "Low"
    telemetry: TelemetrySummary
    hazards: HazardScores
    logged_at: datetime

    model_config = ConfigDict(from_attributes=True)


class VerificationSummary(BaseModel):
    verified_count: int
    mean_brier_score: Optional[float] = None
    calibration_status: str
    requires_retrain: bool = False


class AccuracyMetrics(BaseModel):
    region_key: str
    total_evaluations: int
    average_brier_score: Optional[float] = None
    model_status: str