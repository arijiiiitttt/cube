from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
from app.core.database import get_db
from app.core.security import get_current_user_id
from app.models.model import ModelRegistry

router = APIRouter()


@router.get("/")
async def list_models(
    job_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    q = select(ModelRegistry).where(ModelRegistry.user_id == user_id)
    if job_id:
        q = q.where(ModelRegistry.job_id == job_id)
    q = q.order_by(ModelRegistry.created_at.desc())
    result = await db.execute(q)
    rows   = result.scalars().all()
    return [
        {
            "id":           m.id,
            "name":         m.name,
            "arch":         m.arch,
            "job_id":       m.job_id,
            "status":       m.status,
            "params":       m.params,
            "metrics":      m.metrics,
            "endpoint_url": m.endpoint_url,
            "created_at":   m.created_at,
        }
        for m in rows
    ]


@router.get("/{model_id}")
async def get_model(
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
    return {
        "id":           model.id,
        "name":         model.name,
        "arch":         model.arch,
        "job_id":       model.job_id,
        "status":       model.status,
        "params":       model.params,
        "metrics":      model.metrics or {},
        "model_path":   model.model_path,
        "endpoint_url": model.endpoint_url,
        "api_key":      model.api_key,
        "created_at":   model.created_at,
        "deployed_at":  model.deployed_at,
    }
