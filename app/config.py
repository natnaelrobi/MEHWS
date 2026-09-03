from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AEGIS-core Multi-Hazard Early Warning System"
    DATABASE_URL: str = "sqlite:///./aegis.db"
    REDIS_URL: str = "redis://localhost:6379/0"
    API_V1_STR: str = "/api/v1"
    OPEN_METEO_BASE_URL: str = "https://api.open-meteo.com/v1/forecast"
    OPEN_METEO_SEASONAL_URL: str = "https://seasonal-api.open-meteo.com/v1/seasonal"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()