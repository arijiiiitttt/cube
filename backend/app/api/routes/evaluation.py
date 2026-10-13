from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.core.security import get_current_user_id
from app.models.model import ModelRegistry
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/{model_id}/run")
async def run_evaluation(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    result = await db.execute(
        select(ModelRegistry).where(
            ModelRegistry.id == model_id,
            ModelRegistry.user_id == user_id,
        )
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(404, "Model not found")
    if model.status not in ("ready", "deployed"):
        raise HTTPException(409, f"Model not ready for evaluation (status: {model.status})")

    try:
        from tasks.celery_tasks import evaluate_model_task
        evaluate_model_task.delay(model_id)
    except Exception as e:
        raise HTTPException(503, "Evaluation worker unavailable. Make sure Celery is running.")

    logger.info(f"Evaluation queued for model {model_id}")
    return {"model_id": model_id, "status": "evaluating"}


@router.get("/{model_id}/results")
async def get_eval_results(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    result = await db.execute(
        select(ModelRegistry).where(
            ModelRegistry.id == model_id,
            ModelRegistry.user_id == user_id,
        )
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(404, "Model not found")

    metrics = model.metrics or {}
    return {
        "model_id":            model_id,
        "perplexity":          metrics.get("perplexity"),
        "accuracy":            metrics.get("accuracy"),
        "bleu":                metrics.get("bleu"),
        "repetition_rate":     metrics.get("repetition_rate"),
        "eval_loss":           metrics.get("eval_loss"),
        "grade":               metrics.get("grade"),
        "n_batches_evaluated": metrics.get("n_batches_evaluated"),
        "error":               metrics.get("error"),
    }
