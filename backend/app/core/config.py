from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List
import json


class Settings(BaseSettings):
    APP_NAME: str = "Cube"
    SECRET_KEY: str = "change-me"

    # CORS: can be a JSON array string in .env or a Python list
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:3001"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return [i.strip() for i in v.split(",") if i.strip()]
        return v

    # Database — use postgresql+asyncpg:// for async driver
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@host/cube?sslmode=require"

    # Redis / Celery
    REDIS_URL: str = "redis://redis:6379/0"

    # MinIO object storage
    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_ROOT_USER: str = "minioadmin"
    MINIO_ROOT_PASSWORD: str = "minioadmin123"
    MINIO_BUCKET: str = "cube-models"
    MINIO_SECURE: bool = False

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"

    # Limits
    MAX_FILE_SIZE_MB: int = 50
    MAX_TRAINING_JOBS: int = 2

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
