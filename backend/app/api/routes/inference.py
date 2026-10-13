from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field
from app.core.database import get_db
from app.core.security import get_current_user_id
from app.models.model import ModelRegistry
import httpx, time, os, json

router = APIRouter()
INFERENCE_URL = os.getenv("INFERENCE_SERVICE_URL", "http://inference-service:8001")


class GenerateRequest(BaseModel):
    prompt:      str
    max_tokens:  int   = Field(200, ge=1,   le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p:       float = Field(0.9, ge=0.0, le=1.0)
    top_k:       int   = Field(50,  ge=1,   le=200)
    stream:      bool  = False


async def _get_model_or_404(model_id: str, user_id: str, db: AsyncSession) -> ModelRegistry:
    result = await db.execute(
        select(ModelRegistry).where(ModelRegistry.id == model_id, ModelRegistry.user_id == user_id)
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(404, "Model not found")
    if model.status not in ("ready", "deployed"):
        raise HTTPException(409, f"Model not ready for inference (status: {model.status})")
    return model


@router.post("/{model_id}/generate")
async def generate(
    model_id: str,
    req: GenerateRequest,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    model = await _get_model_or_404(model_id, user_id, db)
    base_model   = model.arch        # model_registry.arch stores the HF base_model id for fine-tunes
    adapter_path = model.model_path  # MinIO prefix where the LoRA adapter was uploaded

    payload = {
        "model_id": model_id,
        "base_model": base_model,
        "adapter_path": adapter_path,
        "prompt": req.prompt,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "stream": req.stream,
    }

    if req.stream:
        return StreamingResponse(_proxy_stream(payload), media_type="text/event-stream")

    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{INFERENCE_URL}/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        raise HTTPException(503, "Inference service unavailable. Make sure it is running.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)
    except Exception as e:
        raise HTTPException(500, str(e))

    return {
        "text":             data.get("text", ""),
        "tokens_generated": data.get("tokens_generated", 0),
        "latency_ms":       round((time.time() - start) * 1000, 2),
        "model_id":         model_id,
    }


async def _proxy_stream(payload: dict):
    """Streams Server-Sent-Events chunks straight through from the inference service."""
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", f"{INFERENCE_URL}/generate", json=payload) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    yield f"data: {json.dumps({'error': body.decode(errors='ignore')})}\n\n"
                    return
                async for chunk in resp.aiter_bytes():
                    yield chunk
    except httpx.ConnectError:
        yield f"data: {json.dumps({'error': 'Inference service unavailable'})}\n\n"
