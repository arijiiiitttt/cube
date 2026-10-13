from sqlalchemy import Column, String, DateTime, JSON, Text
from sqlalchemy.sql import func
import uuid
from app.core.database import Base


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id      = Column(String, nullable=False, index=True)
    name         = Column(String, nullable=False)
    arch         = Column(String, nullable=False)
    job_id       = Column(String, nullable=True)
    status       = Column(String, default="training")   # training | ready | deployed | error
    params       = Column(String, nullable=True)
    model_path   = Column(String, nullable=True)
    config       = Column(JSON, default=dict)
    metrics      = Column(JSON, default=dict)
    endpoint_url = Column(String, nullable=True)
    api_key      = Column(String, nullable=True)
    description  = Column(Text, nullable=True)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())
    deployed_at  = Column(DateTime(timezone=True), nullable=True)
