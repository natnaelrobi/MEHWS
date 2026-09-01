# app/tasks/__init__.py
from app.tasks.hazard_tasks import fetch_and_ingest_all_regions

__all__ = ["fetch_and_ingest_all_regions"]