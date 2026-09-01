from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import settings

# Create PostgreSQL connection engine
engine = create_engine(settings.DATABASE_URL)

# Create a local session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for our models (telemetry.py and prediction.py) to inherit from
Base = declarative_base()

# Dependency to get the DB session in our API routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()