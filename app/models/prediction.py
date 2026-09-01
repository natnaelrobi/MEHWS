import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Path resolution: aegis-core/app/models/prediction.py -> aegis-core/artifacts/
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

# Load artifacts safely at startup
try:
    flood_model = joblib.load(ARTIFACT_DIR / "flood_model_v1.pkl")
    drought_model = joblib.load(ARTIFACT_DIR / "drought_model_v1.pkl")
except Exception:
    flood_model = None
    drought_model = None


def generate_region_prediction(features_dict: dict) -> dict:
    """
    Takes environmental features for a region and generates risk predictions
    matching the HazardPrediction schema.
    """
    # 1. Format inputs as DataFrames to preserve feature names for XGBoost
    flood_df = pd.DataFrame([{
        'rfh': float(features_dict.get('rfh', 0.0)),
        'r1h': float(features_dict.get('r1h', 0.0)),
        'r3h': float(features_dict.get('r3h', 0.0)),
        'slope_mean': float(features_dict.get('slope_mean', 5.0)),
        'soil_moisture_mean': float(features_dict.get('soil_moisture_mean', 400.0)),
        'ndvi_mean': float(features_dict.get('ndvi_mean', 0.4)),
        'dist_to_river_m': float(features_dict.get('dist_to_river_m', 1000.0))
    }])

    drought_df = pd.DataFrame([{
        'rfq': float(features_dict.get('rfq', 50.0)),
        'r3q': float(features_dict.get('r3q', 50.0)),
        'ndvi_mean': float(features_dict.get('ndvi_mean', 0.4)),
        'soil_moisture_mean': float(features_dict.get('soil_moisture_mean', 400.0))
    }])

    # 2. Predict probabilities / scores
    if flood_model and drought_model:
        try:
            flood_prob = float(flood_model.predict_proba(flood_df)[0][1] * 100.0)
            drought_score = float(drought_model.predict(drought_df)[0])
            drought_risk = min(max(drought_score, 0.0), 100.0)
        except Exception:
            # Fallback if model features or ordering mismatch
            flood_prob = min(100.0, (features_dict.get('rfh', 0.0) / 50.0) * 100.0)
            drought_risk = 30.0
    else:
        # Heuristic fallback if artifacts fail to load
        flood_prob = min(100.0, (features_dict.get('rfh', 0.0) / 50.0) * 100.0)
        drought_risk = 30.0

    # Optional landslide risk estimation based on slope and rainfall
    slope = features_dict.get('slope_mean', 5.0)
    rfh = features_dict.get('rfh', 0.0)
    landslide_risk = float(min(max((slope * 1.5) + (rfh * 0.2), 0.0), 100.0))

    # 3. Determine dominant hazard and overall risk level
    risks = {
        "Flood": flood_prob,
        "Drought": drought_risk,
        "Landslide": landslide_risk
    }
    primary_hazard = max(risks, key=risks.get)
    max_risk = risks[primary_hazard]

    if max_risk < 25.0:
        risk_level = "Low"
    elif max_risk < 50.0:
        risk_level = "Moderate"
    elif max_risk < 75.0:
        risk_level = "High"
    else:
        risk_level = "Critical"

    return {
        "predicted_flood_risk": round(flood_prob, 2),
        "predicted_drought_risk": round(drought_risk, 2),
        "predicted_landslide_risk": round(landslide_risk, 2),
        "primary_hazard_type": primary_hazard,
        "overall_risk_level": risk_level
    }