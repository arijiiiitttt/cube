from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.core.security import get_current_user_id
from app.models.dataset import Dataset
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/{dataset_id}/process")
async def process_dataset(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    result  = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.user_id == user_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    if dataset.status == "processing":
        raise HTTPException(409, "Already processing — please wait")
    if dataset.status == "ready":
        raise HTTPException(409, "Already processed")

    dataset.status = "processing"
    await db.commit()

    try:
        from tasks.celery_tasks import process_dataset_task
        process_dataset_task.delay(dataset_id)
    except Exception as e:
        dataset.status = "error"
        await db.commit()
        raise HTTPException(503, "Processing worker unavailable. Make sure Celery is running.")

    logger.info(f"Processing queued for dataset {dataset_id}")
    return {"dataset_id": dataset_id, "status": "processing"}


@router.get("/{dataset_id}/status")
async def dataset_status(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    result  = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id, Dataset.user_id == user_id)
    )
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    return {
        "id":           dataset.id,
        "name":         dataset.name,
        "status":       dataset.status,
        "total_tokens": dataset.total_tokens,
        "train_tokens": dataset.train_tokens,
        "val_tokens":   dataset.val_tokens,
        "vocab_size":   dataset.vocab_size,
        "stats":        dataset.stats or {},
        "raw_bytes":    dataset.raw_bytes,
        "created_at":   dataset.created_at,
    }


@router.get("/")
async def list_datasets(
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    result = await db.execute(
        select(Dataset)
        .where(Dataset.user_id == user_id)
        .order_by(Dataset.created_at.desc())
        .limit(50)
    )
    return result.scalars().all()
