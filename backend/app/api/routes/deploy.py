from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
from app.core.database import get_db
from app.core.security import get_current_user_id
from app.models.model import ModelRegistry
import secrets

router = APIRouter()


@router.post("/{model_id}")
async def deploy_model(
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
        raise HTTPException(409, f"Model cannot be deployed (status: {model.status})")

    api_key      = f"cube_sk_{secrets.token_hex(16)}"
    endpoint_url = f"https://api.cube.dev/inference/{model_id}/generate"

    model.status       = "deployed"
    model.api_key      = api_key
    model.endpoint_url = endpoint_url
    model.deployed_at  = datetime.utcnow()
    await db.commit()

    return {
        "model_id":     model_id,
        "endpoint_url": endpoint_url,
        "api_key":      api_key,
        "status":       "deployed",
    }


@router.get("/{model_id}")
async def get_deploy_info(
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
        "model_id":     model.id,
        "endpoint_url": model.endpoint_url,
        "api_key":      model.api_key,
        "status":       model.status,
        "deployed_at":  model.deployed_at,
    }
