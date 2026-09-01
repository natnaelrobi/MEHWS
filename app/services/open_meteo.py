import requests
from typing import Dict, Any, Optional

REGIONS_COORDS = {
    "addis_ababa": {"lat": 9.03, "lon": 38.74, "name": "Addis Ababa", "pcode": "ET0101", "slope": 14.5},
    "awash_basin": {"lat": 8.98, "lon": 40.17, "name": "Awash Basin", "pcode": "ET0402", "slope": 3.2},
    "dire_dawa": {"lat": 9.59, "lon": 41.86, "name": "Dire Dawa", "pcode": "ET1501", "slope": 18.0}
}

def fetch_weather_forecast(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Fetches 7-day hourly forecast for rainfall and soil moisture."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=rain,soil_moisture_0_to_7cm&timezone=auto"
    )
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException:
        return None