from celery import Celery
from celery.schedules import crontab

celery_app = Celery(
    "aegis_worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    beat_schedule={
        "fetch-weather-every-6-hours": {
            "task": "tasks.fetch_and_ingest_all_regions",
            "schedule": crontab(minute=0, hour="*/6"),
        },
    },
)

celery_app.autodiscover_tasks(["app.tasks"])