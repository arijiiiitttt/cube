from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.core.security import get_current_user_id
from app.core.config import settings
from app.models.job import TrainingJob, JobStatus
from app.schemas.training import (
    TrainingConfig,
    TrainingEstimateRequest,
    estimate_training_time,
)
import uuid, redis, logging

logger = logging.getLogger(__name__)
router  = APIRouter()

# Use Redis URL from settings (works both locally and via Docker)
_redis = redis.from_url(settings.REDIS_URL, decode_responses=False)


@router.post("/estimate")
async def estimate_job(
    req: TrainingEstimateRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Pre-flight estimate shown in the UI before a job is queued. Cheap heuristic, not a profiler run."""
    return estimate_training_time(req)


@router.post("/start")
async def start_training(
    config: TrainingConfig,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    # Cap concurrent jobs per user
    running = await db.execute(
        select(TrainingJob).where(
            TrainingJob.status == JobStatus.running,
            TrainingJob.user_id == user_id,
        )
    )
    if len(running.scalars().all()) >= settings.MAX_TRAINING_JOBS:
        raise HTTPException(429, f"Max {settings.MAX_TRAINING_JOBS} concurrent training jobs")

    job = TrainingJob(
        id=str(uuid.uuid4()),
        name=config.name,
        model_arch=config.base_model,
        dataset_id=config.dataset_id,
        config=config.model_dump(),
        total_epochs=config.epochs,
        status=JobStatus.pending,
        user_id=user_id,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    try:
        from tasks.celery_tasks import train_model_task
        train_model_task.delay(job.id, config.model_dump())
    except Exception as e:
        logger.error(f"Failed to queue training task: {e}")
        job.status    = JobStatus.error
        job.error_msg = "Failed to queue task — is Celery running?"
        await db.commit()
        raise HTTPException(503, "Training worker unavailable. Make sure Celery is running.")

    logger.info(f"Training job {job.id} queued for user {user_id}")
    return {
        "id":            job.id,
        "status":        job.status,
        "name":          job.name,
        "current_epoch": 0,
        "total_epochs":  job.total_epochs,
        "current_step":  0,
        "total_steps":   0,
    }


@router.get("/{job_id}/status")
async def training_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    result = await db.execute(
        select(TrainingJob).where(TrainingJob.id == job_id, TrainingJob.user_id == user_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "id":            job.id,
        "status":        job.status,
        "name":          job.name,
        "model_arch":    job.model_arch,
        "current_epoch": job.current_epoch,
        "total_epochs":  job.total_epochs,
        "current_step":  job.current_step,
        "total_steps":   job.total_steps,
        "train_loss":    job.train_loss,
        "val_loss":      job.val_loss,
        "lr":            job.lr,
        "model_path":    job.model_path,
        "error_msg":     job.error_msg,
    }


@router.post("/{job_id}/stop")
async def stop_training(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    result = await db.execute(
        select(TrainingJob).where(TrainingJob.id == job_id, TrainingJob.user_id == user_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status not in (JobStatus.running, JobStatus.pending):
        raise HTTPException(409, f"Cannot stop job with status: {job.status}")

    job.status = JobStatus.paused
    await db.commit()
    _redis.set(f"stop:{job_id}", "1", ex=3600)
    logger.info(f"Stop signal sent for job {job_id}")
    return {"status": "stopped", "job_id": job_id}


@router.get("/{job_id}/logs")
async def training_logs(
    job_id: str,
    user_id: str = Depends(get_current_user_id),
):
    try:
        logs = _redis.lrange(f"logs:{job_id}", 0, -1)
        return {"logs": [l.decode() if isinstance(l, bytes) else l for l in logs]}
    except Exception as e:
        logger.error(f"Redis error fetching logs: {e}")
        return {"logs": [], "warning": "Log service unavailable"}


@router.get("/")
async def list_jobs(
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    result = await db.execute(
        select(TrainingJob)
        .where(TrainingJob.user_id == user_id)
        .order_by(TrainingJob.created_at.desc())
        .limit(50)
    )
    jobs = result.scalars().all()
    return [
        {
            "id":            j.id,
            "name":          j.name,
            "status":        j.status,
            "model_arch":    j.model_arch,
            "current_epoch": j.current_epoch,
            "total_epochs":  j.total_epochs,
            "train_loss":    j.train_loss,
            "created_at":    j.created_at,
        }
        for j in jobs
    ]
