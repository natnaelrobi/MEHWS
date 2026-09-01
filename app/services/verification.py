import requests
from datetime import datetime
from sqlalchemy.orm import Session
from app.database import HazardPrediction
from app.services.open_meteo import REGIONS_COORDS
from app.services.retrainer import retrain_models_from_real_world_data

def fetch_historical_actuals(lat: float, lon: float, date_str: str):
    """Pulls actual historical daily rainfall from Open-Meteo Archive for verification."""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&daily=rain_sum&timezone=auto"
    )
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            rain_list = data.get("daily", {}).get("rain_sum", [])
            if rain_list:
                return rain_list[0]
    except requests.exceptions.RequestException:
        pass
    return None

def verify_pending_predictions(db: Session, region_key: str = None):
    """
    Scans hazard_predictions for past target dates missing verification, 
    fetches ground truth, calculates Brier Error Scores, and triggers 
    automated retraining if performance drift is detected.
    """
    now = datetime.utcnow()
    
    # Query unverified records whose target dates have passed
    query = db.query(HazardPrediction).filter(
        HazardPrediction.target_date <= now,
        HazardPrediction.is_verified == False
    )
    if region_key:
        query = query.filter(HazardPrediction.region_key == region_key)
        
    pending_records = query.all()
    verified_count = 0
    brier_scores = []

    for record in pending_records:
        coords = REGIONS_COORDS.get(record.region_key)
        if not coords:
            continue

        target_date_str = record.target_date.strftime("%Y-%m-%d")
        actual_rain = fetch_historical_actuals(coords["lat"], coords["lon"], target_date_str)

        if actual_rain is not None:
            # Ground truth: Flood occurred if 24h rainfall exceeded 35mm threshold
            flood_occurred = actual_rain >= 35.0
            
            # Normalize predicted risk percentage (0-100%) to probability (0.0 - 1.0)
            predicted_prob = record.predicted_flood_risk / 100.0
            actual_binary = 1.0 if flood_occurred else 0.0

            # Compute Brier Score
            brier_score = round((predicted_prob - actual_binary) ** 2, 4)

            # Update PostgreSQL database record
            record.actual_realized_rainfall = actual_rain
            record.actual_flood_occurred = flood_occurred
            record.brier_error_score = brier_score
            record.is_verified = True

            verified_count += 1
            brier_scores.append(brier_score)

    db.commit()

    mean_brier = round(sum(brier_scores) / len(brier_scores), 4) if brier_scores else None
    requires_retrain = (mean_brier is not None) and (mean_brier > 0.15)

    retrain_result = None
    if requires_retrain:
        print("⚠️ Model performance drift detected. Executing automated self-healing retraining loop...")
        retrain_result = retrain_models_from_real_world_data()

    return {
        "verified_count": verified_count,
        "mean_brier_score": mean_brier,
        "calibration_status": "Drift Detected - Models Retrained" if requires_retrain else "Well Calibrated",
        "requires_retrain": requires_retrain,
        "retrain_details": retrain_result
    }