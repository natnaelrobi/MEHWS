
from datetime import datetime, timedelta
from app.celery_app import celery_app
from app.database import SessionLocal
from app.models.telemetry import WeatherTelemetry
from app.database import HazardPrediction
from app.services.open_meteo import REGIONS_COORDS, fetch_weather_forecast
from app.services.risk_calculator import compute_multi_hazard_scores

@celery_app.task(name="tasks.fetch_and_ingest_all_regions")
def fetch_and_ingest_all_regions():
    db = SessionLocal()
    try:
        processed_regions = []
        for region_key, coords in REGIONS_COORDS.items():
            raw_data = fetch_weather_forecast(coords["lat"], coords["lon"])
            if not raw_data:
                continue
            
            hourly_rain = raw_data.get("hourly", {}).get("rain", [])
            scores = compute_multi_hazard_scores(
                hourly_rain=hourly_rain,
                soil_moisture=0.25,
                slope_degrees=coords.get("slope")
            )
            
            # Log Telemetry Snapshot
            telemetry = WeatherTelemetry(
                region_key=region_key,
                pcode=coords.get("pcode"),
                timestamp=datetime.utcnow(),
                rain_24h=scores["rain_24h"],
                rain_72h=scores["rain_72h"],
                rain_30d=scores["rain_30d"],
                flash_flood_risk=scores["flash_flood_risk"],
                drought_risk=scores["drought_risk"]
            )
            db.add(telemetry)
            
            # Stage Prediction Target
            primary_hazard = "Flood" if scores["flash_flood_risk"] > scores["drought_risk"] else "Drought"
            max_prob = max(scores["flash_flood_risk"], scores["drought_risk"])
            risk_level = "High" if max_prob > 70 else ("Moderate" if max_prob > 35 else "Low")
            
            prediction = HazardPrediction(
                region_key=region_key,
                pcode=coords.get("pcode"),
                predicted_at=datetime.utcnow(),
                target_date=datetime.utcnow() + timedelta(days=7),
                predicted_flood_risk=scores["flash_flood_risk"],
                predicted_drought_risk=scores["drought_risk"],
                primary_hazard_type=primary_hazard,
                overall_risk_level=risk_level
            )
            db.add(prediction)
            processed_regions.append(region_key)
            
        db.commit()
        return {"status": "success", "ingested_regions": processed_regions}
    finally:
        db.close()