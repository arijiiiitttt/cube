from sqlalchemy import Column, String, Integer, BigInteger, DateTime, JSON
from sqlalchemy.sql import func
import uuid
from app.core.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id              = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id         = Column(String, nullable=False, index=True)
    name            = Column(String, nullable=False)
    status          = Column(String, default="pending")   # pending | processing | ready | error
    file_paths      = Column(JSON, default=list)
    total_tokens    = Column(Integer, nullable=True)
    train_tokens    = Column(Integer, nullable=True)
    val_tokens      = Column(Integer, nullable=True)
    vocab_size      = Column(Integer, nullable=True)
    tokenizer_path  = Column(String, nullable=True)
    stats           = Column(JSON, default=dict)
    raw_bytes       = Column(BigInteger, default=0)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    updated_at      = Column(DateTime(timezone=True), onupdate=func.now())
