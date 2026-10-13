"""
Cube — Inference Service

Loads a fine-tuned LoRA adapter from MinIO, attaches it to its HF base model,
and serves generation requests (including token streaming). Adapters are
small (a few MB-~100s of MB), so loading per-model is cheap; the base model
itself is cached process-wide and shared across adapters when multiple jobs
fine-tuned the same base — this avoids reloading multi-GB base weights for
every request.
"""
import os, time, logging, json, tempfile
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import torch
from minio import Minio
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app    = FastAPI(title="Cube Inference Service", version="0.2.0")
BUCKET = os.getenv("MINIO_BUCKET", "cube-models")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Separate caches: base models are large and shared across adapters fine-tuned
# from the same checkpoint; adapters are cheap and one-per-(base, path) pair.
_base_cache: dict = {}     # base_model_id -> AutoModelForCausalLM
_adapter_cache: dict = {}  # f"{base_model}::{adapter_path}" -> {model, tokenizer, base_model}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)


def get_minio() -> Minio:
    return Minio(
        os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.getenv("MINIO_ROOT_USER", "minioadmin"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin123"),
        secure=False,
    )


def _load_base(base_model: str):
    if base_model in _base_cache:
        return _base_cache[base_model]
    logger.info(f"Loading base model {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    _base_cache[base_model] = model
    return model


def load_adapter(model_id: str, base_model: str, adapter_path: str) -> dict:
    cache_key = f"{base_model}::{adapter_path}"
    if cache_key in _adapter_cache:
        return _adapter_cache[cache_key]

    logger.info(f"Loading adapter for model {model_id} from MinIO ({adapter_path})...")
    client = get_minio()

    with tempfile.TemporaryDirectory() as tmp:
        objects = list(client.list_objects(BUCKET, prefix=f"{adapter_path}/", recursive=True))
        if not objects:
            raise HTTPException(404, f"Adapter not found in storage at {adapter_path}")
        for obj in objects:
            rel = os.path.relpath(obj.object_name, adapter_path)
            local_f = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(local_f), exist_ok=True)
            client.fget_object(BUCKET, obj.object_name, local_f)

        tokenizer = AutoTokenizer.from_pretrained(tmp)
        base = _load_base(base_model)
        model = PeftModel.from_pretrained(base, tmp, adapter_name=model_id)
        model.eval()

    _adapter_cache[cache_key] = {"model": model, "tokenizer": tokenizer, "base_model": base_model}
    logger.info(f"Adapter {model_id} loaded on base {base_model} (device={device})")
    return _adapter_cache[cache_key]


class GenerateRequest(BaseModel):
    model_id: str
    base_model: str = Field(..., description="HF hub ID of the base model this adapter was fine-tuned from.")
    adapter_path: str = Field(..., description="MinIO prefix of the LoRA adapter, e.g. models/<job_id>/adapter.")
    prompt: str
    max_tokens: int = Field(200, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.1, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=1, le=500)
    stream: bool = False


def _build_inputs(tokenizer, prompt: str):
    enc = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        enc = {k: v.to(device) for k, v in enc.items()}
    return enc


@app.post("/generate")
async def generate(req: GenerateRequest):
    try:
        ctx = load_adapter(req.model_id, req.base_model, req.adapter_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"Model load failed: {e}")

    model, tokenizer = ctx["model"], ctx["tokenizer"]
    enc = _build_inputs(tokenizer, req.prompt)

    gen_kwargs = dict(
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        do_sample=req.temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )

    if req.stream:
        return StreamingResponse(_stream_tokens(model, tokenizer, enc, gen_kwargs), media_type="text/event-stream")

    start = time.time()
    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)
    latency = (time.time() - start) * 1000

    new_ids = out[0, enc["input_ids"].shape[1]:]
    generated = tokenizer.decode(new_ids, skip_special_tokens=True)

    return {
        "text": generated,
        "prompt": req.prompt,
        "tokens_generated": len(new_ids),
        "latency_ms": round(latency, 2),
        "model_id": req.model_id,
        "device": str(device),
    }


async def _stream_tokens(model, tokenizer, enc, gen_kwargs) -> AsyncGenerator[str, None]:
    """Server-Sent-Events stream of generated text chunks, for the inference playground."""
    from threading import Thread

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = Thread(target=model.generate, kwargs={**enc, **gen_kwargs, "streamer": streamer})
    thread.start()

    for chunk in streamer:
        yield f"data: {json.dumps({'text': chunk})}\n\n"
    yield "data: [DONE]\n\n"
    thread.join()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(device),
        "loaded_adapters": list(_adapter_cache.keys()),
        "loaded_base_models": list(_base_cache.keys()),
        "cuda_available": torch.cuda.is_available(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
