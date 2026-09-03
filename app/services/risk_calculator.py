import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

ARTIFACT_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"

try:
    flood_model = joblib.load(ARTIFACT_DIR / "flood_model_v1.pkl")
    drought_model = joblib.load(ARTIFACT_DIR / "drought_model_v1.pkl")
except Exception:
    flood_model = None
    drought_model = None

def compute_multi_hazard_scores(
    hourly_rain: List[float],
    soil_moisture: float = 400.0,
    historical_30d_avg_rain: float = 120.0,
    slope_degrees: Optional[float] = 10.0,
    ndvi_mean: float = 0.4,
    dist_to_river_m: float = 1000.0,
    enable_landslide: bool = False
) -> Dict[str, float]:
    
    # 1. Derive short-term, medium-term, and long-term precipitation metrics
    rfh = float(sum(hourly_rain[:24])) if len(hourly_rain) >= 24 else (float(sum(hourly_rain)) if hourly_rain else 0.0)
    rain_72h = float(sum(hourly_rain[:72])) if len(hourly_rain) >= 72 else (rfh * 3.0)
    r1h = float(sum(hourly_rain[:720])) if len(hourly_rain) >= 720 else (rain_72h * 10.0)
    r3h = r1h * 3.0
    
    slope = slope_degrees if slope_degrees is not None else 10.0

    # 2. Build Flood Model Feature DataFrame (Matches trained XGBoost column names)
    flood_features = pd.DataFrame([{
        'rfh': rfh,
        'r1h': r1h,
        'r3h': r3h,
        'slope_mean': slope,
        'soil_moisture_mean': soil_moisture,
        'ndvi_mean': ndvi_mean,
        'dist_to_river_m': dist_to_river_m
    }])

    # 3. Build Drought Model Feature DataFrame
    rfq = max(0.0, historical_30d_avg_rain - rfh)
    r3q = max(0.0, (historical_30d_avg_rain * 3.0) - r1h)

    drought_features = pd.DataFrame([{
        'rfq': rfq,
        'r3q': r3q,
        'ndvi_mean': ndvi_mean,
        'soil_moisture_mean': soil_moisture
    }])

    # 4. Generate Predictions using Binary Artifacts
    if flood_model and drought_model:
        try:
            flash_flood_risk = float(flood_model.predict_proba(flood_features)[0][1] * 100.0)
            raw_drought = float(drought_model.predict(drought_features)[0])
            drought_risk = float(max(0.0, min(100.0, raw_drought)))
        except Exception:
            # Fallback if model structure or pandas column ordering throws error
            flash_flood_risk = min(100.0, (rfh / 50.0) * 100.0)
            drought_risk = 30.0
    else:
        # Heuristic fallback if .pkl files aren't found
        flash_flood_risk = min(100.0, (rfh / 50.0) * 100.0)
        drought_risk = 30.0

def compute_multi_hazard_scores(
    hourly_rain: List[float],
    daily_seasonal_rain: Optional[List[float]] = None, # New parameter for multi-month data
    soil_moisture: float = 400.0,
    historical_30d_avg_rain: float = 120.0,
    slope_degrees: Optional[float] = 10.0,
    ndvi_mean: float = 0.4,
    dist_to_river_m: float = 1000.0,
    enable_landslide: bool = False
) -> Dict[str, float]:
    
    # 1. Derive short-term and long-term precipitation metrics
    rfh = float(sum(hourly_rain[:24])) if len(hourly_rain) >= 24 else (float(sum(hourly_rain)) if hourly_rain else 0.0)
    rain_72h = float(sum(hourly_rain[:72])) if len(hourly_rain) >= 72 else (rfh * 3.0)
    
    # If seasonal daily data is passed (e.g., 90 to 180 days ahead), compute true long-term totals
    if daily_seasonal_rain and len(daily_seasonal_rain) > 0:
        r1h = float(sum(daily_seasonal_rain[:30])) # 30-day cumulative precipitation
        r_90d = float(sum(daily_seasonal_rain[:90])) # 90-day multi-month cumulative
    else:
        r1h = float(sum(hourly_rain[:720])) if len(hourly_rain) >= 720 else (rain_72h * 10.0)
        r_90d = r1h * 3.0

    slope = slope_degrees if slope_degrees is not None else 10.0

    # 2. Build Flood Model Feature DataFrame
    flood_features = pd.DataFrame([{
        'rfh': rfh,
        'r1h': r1h,
        'r3h': r1h * 3.0,
        'slope_mean': slope,
        'soil_moisture_mean': soil_moisture,
        'ndvi_mean': ndvi_mean,
        'dist_to_river_m': dist_to_river_m
    }])

    # 3. Build Drought Model Feature DataFrame utilizing extended horizons
    rfq = max(0.0, historical_30d_avg_rain - rfh)
    r3q = max(0.0, (historical_30d_avg_rain * 3.0) - r_90d)

    drought_features = pd.DataFrame([{
        'rfq': rfq,
        'r3q': r3q,
        'ndvi_mean': ndvi_mean,
        'soil_moisture_mean': soil_moisture
    }])

    # 4. Generate Predictions using Binary Artifacts
    if flood_model and drought_model:
        try:
            flash_flood_risk = float(flood_model.predict_proba(flood_features)[0][1] * 100.0)
            raw_drought = float(drought_model.predict(drought_features)[0])
            drought_risk = float(max(0.0, min(100.0, raw_drought)))
        except Exception:
            flash_flood_risk = min(100.0, (rfh / 50.0) * 100.0)
            drought_risk = 30.0
    else:
        flash_flood_risk = min(100.0, (rfh / 50.0) * 100.0)
        drought_risk = 30.0

    # 5. Landslide Heuristic Calculation
    landslide_risk = None
    if enable_landslide and slope_degrees is not None:
        slope_factor = min(1.0, slope / 35.0)
        landslide_risk = round(min(100.0, ((rain_72h / 100.0) * 0.6 + slope_factor * 0.4) * 100.0), 1)

    return {
        "rain_24h": round(rfh, 2),
        "rain_72h": round(rain_72h, 2),
        "rain_30d": round(r1h, 2),
        "flash_flood_risk": round(flash_flood_risk, 1),
        "drought_risk": round(drought_risk, 1),
        "landslide_risk": landslide_risk
    }

