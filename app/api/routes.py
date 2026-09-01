from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db, HazardPrediction
from app.models.telemetry import WeatherTelemetry
from app.models.prediction import generate_region_prediction
from app.services.open_meteo import REGIONS_COORDS, fetch_weather_forecast
from app.services.risk_calculator import compute_multi_hazard_scores
from app.services.verification import verify_pending_predictions
from app.tasks.hazard_tasks import fetch_and_ingest_all_regions

router = APIRouter(prefix="/api/v1", tags=["Multi-Hazard Ingestion"])

@router.post("/ingest/{region_key}")
def ingest_region_hazards(
    region_key: str, 
    include_landslide: bool = False,
    db: Session = Depends(get_db)
):
    if region_key not in REGIONS_COORDS:
        raise HTTPException(status_code=404, detail=f"Region key '{region_key}' not found in registry")

    coords = REGIONS_COORDS[region_key]
    raw_data = fetch_weather_forecast(coords["lat"], coords["lon"])
    if not raw_data:
        raise HTTPException(status_code=502, detail="Open-Meteo API payload unavailable or timed out")

    hourly_rain = raw_data.get("hourly", {}).get("rain", [])
    
    # 1. Compute multi-hazard probabilities
    scores = compute_multi_hazard_scores(
        hourly_rain=hourly_rain,
        soil_moisture=0.20,
        slope_degrees=coords.get("slope"),
        enable_landslide=include_landslide
    )

    # 2. Persist weather telemetry baseline record
    telemetry = WeatherTelemetry(
        region_key=region_key,
        pcode=coords.get("pcode"),
        timestamp=datetime.utcnow(),
        rain_24h=scores["rain_24h"],
        rain_72h=scores["rain_72h"],
        rain_30d=scores["rain_30d"],
        flash_flood_risk=scores["flash_flood_risk"],
        drought_risk=scores["drought_risk"],
        landslide_risk=scores["landslide_risk"]
    )
    db.add(telemetry)

    # 3. Categorize primary hazard and severity classification
    primary_hazard = "Flood" if scores["flash_flood_risk"] > scores["drought_risk"] else "Drought"
    max_prob = max(scores["flash_flood_risk"], scores["drought_risk"])
    
    if max_prob > 70.0:
        risk_level = "High"
    elif max_prob > 35.0:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    # 4. Stage predictive assessment
    prediction = HazardPrediction(
        region_key=region_key,
        pcode=coords.get("pcode"),
        predicted_at=datetime.utcnow(),
        target_date=datetime.utcnow() + timedelta(days=7),
        predicted_flood_risk=scores["flash_flood_risk"],
        predicted_drought_risk=scores["drought_risk"],
        predicted_landslide_risk=scores["landslide_risk"],
        primary_hazard_type=primary_hazard,
        overall_risk_level=risk_level
    )
    db.add(prediction)
    db.commit()

    return {
        "status": "success",
        "region": coords["name"],
        "primary_hazard_type": primary_hazard,
        "overall_risk_level": risk_level,
        "hazard_scores": {
            "flash_flood_risk_pct": scores["flash_flood_risk"],
            "drought_risk_pct": scores["drought_risk"],
            "landslide_risk_pct": scores["landslide_risk"]
        }
    }


@router.post("/verify/run")
def trigger_verification(
    region_key: Optional[str] = None, 
    db: Session = Depends(get_db)
):
    """Triggers ground-truth audit against Open-Meteo Historical Archive."""
    results = verify_pending_predictions(db, region_key=region_key)
    return results


@router.get("/metrics/accuracy/{region_key}")
def get_model_accuracy(region_key: str, db: Session = Depends(get_db)):
    """Retrieves mean Brier score and model calibration health for a region."""
    stats = (
        db.query(
            func.count(HazardPrediction.id).label("total"),
            func.avg(HazardPrediction.brier_error_score).label("avg_brier"),
        )
        .filter(
            HazardPrediction.region_key == region_key,
            HazardPrediction.brier_error_score.isnot(None),
        )
        .first()
    )

    avg_brier = round(stats.avg_brier, 4) if stats and stats.avg_brier else None

    status_str = "Well Calibrated"
    if avg_brier and avg_brier > 0.15:
        status_str = "Recalibration Triggered"

    return {
        "region_key": region_key,
        "total_verified_evaluations": (stats.total if stats else 0) or 0,
        "mean_brier_score": avg_brier,
        "calibration_status": status_str,
    }


@router.post("/tasks/trigger-ingestion", tags=["Automation Pipeline"])
def trigger_celery_ingestion():
    """Manually dispatches the ingestion background job to Celery/Redis."""
    task = fetch_and_ingest_all_regions.delay()
    return {
        "status": "success",
        "message": "Background ingestion task dispatched successfully.",
        "task_id": task.id
    }