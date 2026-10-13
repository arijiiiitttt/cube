from sqlalchemy import Column, String, Integer, Float, DateTime, Enum, JSON, Text
from sqlalchemy.sql import func
import enum
import uuid
from app.core.database import Base


class JobStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    paused  = "paused"
    done    = "done"
    error   = "error"


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id            = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id       = Column(String, nullable=False, index=True)
    name          = Column(String, nullable=False)
    status        = Column(Enum(JobStatus), default=JobStatus.pending, nullable=False)
    model_arch    = Column(String, nullable=False)
    dataset_id    = Column(String, nullable=True)
    config        = Column(JSON, nullable=False, default=dict)
    current_epoch = Column(Integer, default=0)
    total_epochs  = Column(Integer, default=3)
    current_step  = Column(Integer, default=0)
    total_steps   = Column(Integer, default=0)
    train_loss    = Column(Float, nullable=True)
    val_loss      = Column(Float, nullable=True)
    lr            = Column(Float, nullable=True)
    model_path    = Column(String, nullable=True)
    error_msg     = Column(Text, nullable=True)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
    updated_at    = Column(DateTime(timezone=True), onupdate=func.now())
    finished_at   = Column(DateTime(timezone=True), nullable=True)
