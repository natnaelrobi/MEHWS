import requests
from typing import Dict, Any, Optional
from config import settings  # Import the settings to use the URLs

REGIONS_COORDS = {
    "addis_ababa": {"lat": 9.03, "lon": 38.74, "name": "Addis Ababa", "pcode": "ET0101", "slope": 14.5},
    "awash_basin": {"lat": 8.98, "lon": 40.17, "name": "Awash Basin", "pcode": "ET0402", "slope": 3.2},
    "dire_dawa": {"lat": 9.59, "lon": 41.86, "name": "Dire Dawa", "pcode": "ET1501", "slope": 18.0}
}

def fetch_tactical_weather(lat: float, lon: float, days: int = 16) -> Optional[Dict[str, Any]]:
    """
    Fetches up to 16-day hourly forecast for rainfall and soil moisture.
    Used for rapid-onset hazard detection like flash floods.
    """
    url = (
        f"{settings.OPEN_METEO_BASE_URL}?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=rain,soil_moisture_0_to_7cm"
        f"&forecast_days={days}&timezone=auto"
    )
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching tactical weather: {e}")
        return None


def fetch_seasonal_climate(lat: float, lon: float, days: int = 180) -> Optional[Dict[str, Any]]:
    """
    Fetches up to 6-month (180 days) daily forecast for precipitation and maximum temperature.
    Used for slow-moving hazard early warning systems like agricultural droughts.
    """
    url = (
        f"{settings.OPEN_METEO_SEASONAL_URL}?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=precipitation_sum,temperature_2m_max"
        f"&forecast_days={days}&timezone=auto"
    )
    try:
        res = requests.get(url, timeout=15) # Slightly longer timeout for seasonal model data
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching seasonal climate: {e}")
        return None