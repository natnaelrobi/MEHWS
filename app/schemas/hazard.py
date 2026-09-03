from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict
import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path

# --- Pydantic Schemas ---

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


# --- Core Risk Calculation & Inference Logic ---

class RiskCalculator:
    def __init__(self):
        # Dynamically locate artifacts directory relative to this file
        self.artifacts_dir = Path(__file__).resolve().parent.parent.parent / "artifacts"
        self.flood_model = self._load_model("aegis_flood_ensemble_model.pkl")
        self.drought_model = self._load_model("aegis_drought_ensemble_model.pkl")

    def _load_model(self, filename: str):
        """Safely loads serialized machine learning models from the artifacts folder."""
        path = self.artifacts_dir / filename
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as e:
                print(f"Warning: Could not load model {filename}: {e}")
        return None

    def calculate_telemetry_and_risks(self, woreda_meta: Dict[str, Any], tactical_weather: dict, seasonal_climate: dict) -> HazardScores:
        """
        Extracts features from Open-Meteo responses and runs model inference 
        for both flash floods (short term) and agricultural droughts (long term months ahead).
        """
        # 1. Extract short-term telemetry for Floods from tactical forecast
        hourly = tactical_weather.get("hourly", {})
        rain_list = hourly.get("rain", [0.0])
        soil_moisture_list = hourly.get("soil_moisture_0_to_7cm", [0.0])

        rfh = float(sum(rain_list[:24])) if len(rain_list) >= 24 else float(sum(rain_list))
        rain_72h = float(sum(rain_list[:72])) if len(rain_list) >= 72 else (rfh * 3.0)
        r1h = float(sum(rain_list[:720])) if len(rain_list) >= 720 else (rain_72h * 10.0)
        r3h = r1h * 3.0
        soil_mean = float(np.mean(soil_moisture_list)) if soil_moisture_list else 400.0

        # 2. Extract long-term telemetry for Droughts from seasonal climate forecast (months ahead)
        daily_climate = seasonal_climate.get("daily", {})
        precip_sum_list = daily_climate.get("precipitation_sum", [0.0])
        rain_30d = sum(precip_sum_list[:30]) if len(precip_sum_list) >= 30 else sum(precip_sum_list)
        rain_90d = sum(precip_sum_list[:90]) if len(precip_sum_list) >= 90 else sum(precip_sum_list)

        # 3. Extract spatial metadata variables
        slope = woreda_meta.get("slope", 10.0)
        ndvi = woreda_meta.get("ndvi_mean", 0.4)
        dist_to_river = woreda_meta.get("dist_to_river_m", 1000.0)
        historical_30d_avg = woreda_meta.get("historical_30d_avg_rain", 120.0)

        # 4. Model Inference with Correct Pipeline Column Schema
        if self.flood_model:
            flood_features = pd.DataFrame([{
                'rfh': rfh,
                'r1h': r1h,
                'r3h': r3h,
                'slope_mean': slope,
                'soil_moisture_mean': soil_mean,
                'ndvi_mean': ndvi,
                'dist_to_river_m': dist_to_river
            }])
            try:
                flood_prob = float(self.flood_model.predict_proba(flood_features)[0][1] * 100.0)
            except Exception:
                flood_prob = min(100.0, (rfh / 50.0) * 100.0)
        else:
            flood_prob = min(100.0, (rfh / 50.0) * 100.0)

        if self.drought_model:
            rfq = max(0.0, historical_30d_avg - rfh)
            r3q = max(0.0, (historical_30d_avg * 3.0) - rain_90d)
            
            drought_features = pd.DataFrame([{
                'rfq': rfq,
                'r3q': r3q,
                'ndvi_mean': ndvi,
                'soil_moisture_mean': soil_mean
            }])
            try:
                # Handle classifiers (predict_proba) vs regressors (predict) safely
                if hasattr(self.drought_model, "predict_proba"):
                    drought_prob = float(self.drought_model.predict_proba(drought_features)[0][1] * 100.0)
                else:
                    drought_prob = float(max(0.0, min(100.0, self.drought_model.predict(drought_features)[0])))
            except Exception:
                drought_prob = max(0.0, min(100.0, 100.0 - (rain_90d / 2.0)))
        else:
            drought_prob = max(0.0, min(100.0, 100.0 - (rain_90d / 2.0)))

        # 5. Landslide Heuristic Calculation
        slope_factor = min(1.0, slope / 35.0)
        landslide_prob = round(min(100.0, ((rain_72h / 100.0) * 0.6 + slope_factor * 0.4) * 100.0), 1)

        return HazardScores(
            flash_flood_risk_pct=round(flood_prob, 2),
            drought_risk_pct=round(drought_prob, 2),
            landslide_risk_pct=round(landslide_prob, 2)
        )

risk_calculator = RiskCalculator()