from fastapi import FastAPI
from app.config import settings
from app.database import engine, Base
from app.api.routes import router

# Import models to ensure SQLAlchemy registers them before creating tables
from app.models import telemetry, prediction 

# Initialize database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Backend API for Aegis Multi-Hazard Early Warning System"
)

# Prefix is already defined in the router itself
app.include_router(router)

@app.get("/")
def read_root():
    return {"status": "online", "system": settings.PROJECT_NAME}