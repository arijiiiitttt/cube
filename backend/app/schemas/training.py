"""
Training configuration schema and time/VRAM estimation.

This is the core contract between the frontend training form, the FastAPI
backend, and the training-service worker. All speed-related knobs (QLoRA,
quantization, attention backend, precision, optimizer) live here so the UI,
API validation, and the actual HF Trainer setup stay in sync.
"""
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional


# Rough param counts (in billions) for HF hub IDs we support out of the box.
# Used only for the pre-flight time/VRAM estimate — not for training itself.
KNOWN_MODEL_SIZES_B = {
    "meta-llama/Llama-3.2-1B-Instruct": 1.0,
    "meta-llama/Llama-3.2-3B-Instruct": 3.0,
    "meta-llama/Llama-3.1-8B-Instruct": 8.0,
    "mistralai/Mistral-7B-Instruct-v0.3": 7.0,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.5,
    "Qwen/Qwen2.5-7B-Instruct": 7.0,
    "google/gemma-2-2b-it": 2.0,
    "microsoft/Phi-3.5-mini-instruct": 3.8,
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 1.1,
}

TARGET_MODULE_PRESETS = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
}


class TrainingConfig(BaseModel):
    # ── Identity ─────────────────────────────────────────────
    name: str = "my-finetune"
    base_model: str = Field(
        "meta-llama/Llama-3.2-1B-Instruct",
        description="Hugging Face hub model ID to fine-tune.",
    )
    dataset_id: str

    # ── Core hyperparameters ────────────────────────────────
    epochs: int = Field(3, ge=1, le=20)
    learning_rate: float = Field(2e-4, ge=1e-6, le=1e-2)
    warmup_ratio: float = Field(0.03, ge=0.0, le=0.3)
    max_seq_len: int = Field(1024, ge=64, le=8192)
    grad_clip: float = Field(1.0, ge=0.0, le=10.0)

    # ── Batching / throughput ───────────────────────────────
    per_device_batch_size: int = Field(4, ge=1, le=128)
    gradient_accumulation_steps: int = Field(4, ge=1, le=128)
    pack_sequences: bool = Field(
        True, description="Pack multiple short examples into one max_seq_len block to cut padding waste."
    )
    dataloader_num_workers: int = Field(2, ge=0, le=16)

    # ── Quantization & precision (speed/memory) ─────────────
    load_in_4bit: bool = Field(True, description="QLoRA: load base model in 4-bit (bitsandbytes NF4).")
    bnb_double_quant: bool = True
    bnb_compute_dtype: Literal["bf16", "fp16"] = "bf16"
    mixed_precision: Literal["bf16", "fp16", "no"] = "bf16"

    # ── LoRA / PEFT ──────────────────────────────────────────
    use_lora: bool = Field(True, description="Disable only for a (slow, memory-heavy) full fine-tune.")
    lora_r: int = Field(16, ge=1, le=256)
    lora_alpha: int = Field(32, ge=1, le=512)
    lora_dropout: float = Field(0.05, ge=0.0, le=0.5)
    lora_target_modules: Optional[list[str]] = Field(
        None, description="Defaults to the model family's attention+MLP projections if omitted."
    )

    # ── Optimizer / kernels ──────────────────────────────────
    optimizer: Literal["paged_adamw_8bit", "adamw_torch", "adamw_torch_fused"] = "paged_adamw_8bit"
    use_flash_attention_2: bool = Field(True, description="Falls back to SDPA automatically if unsupported.")
    use_torch_compile: bool = Field(True, description="torch.compile() the model; adds ~30-60s warmup.")
    use_unsloth: bool = Field(
        False, description="Experimental: route through Unsloth's patched kernels for 1.5-2x extra speedup."
    )
    use_gradient_checkpointing: bool = Field(True, description="Trade ~20% speed for large VRAM savings.")

    @model_validator(mode="after")
    def _validate_combo(self) -> "TrainingConfig":
        if self.use_unsloth and not self.use_lora:
            raise ValueError("use_unsloth requires use_lora=True (Unsloth only supports PEFT fine-tuning).")
        if not self.use_lora and self.load_in_4bit:
            raise ValueError(
                "load_in_4bit without use_lora means a frozen 4-bit model with no trainable adapters. "
                "Either enable use_lora or disable load_in_4bit for a full fine-tune."
            )
        return self


class TrainingEstimateRequest(BaseModel):
    base_model: str
    dataset_total_tokens: int = Field(..., ge=1)
    epochs: int = Field(3, ge=1, le=20)
    per_device_batch_size: int = Field(4, ge=1, le=128)
    gradient_accumulation_steps: int = Field(4, ge=1, le=128)
    max_seq_len: int = Field(1024, ge=64, le=8192)
    load_in_4bit: bool = True
    use_lora: bool = True
    gpu_name: Literal["A100-80GB", "A10G-24GB", "L4-24GB", "T4-16GB", "RTX4090-24GB"] = "A10G-24GB"


# Effective tokens/sec throughput estimates per (GPU, billions-of-params, QLoRA?) —
# calibrated from public fine-tuning benchmarks (HF/Unsloth blog posts), not measured
# in this repo. Treat as a ballpark for the pre-flight estimate, not a guarantee.
_GPU_TOK_S_PER_B_PARAMS = {
    "A100-80GB": 900.0,
    "A10G-24GB": 280.0,
    "L4-24GB": 220.0,
    "T4-16GB": 90.0,
    "RTX4090-24GB": 380.0,
}
_FULL_FT_PENALTY = 0.4  # full fine-tunes are slower per-token than QLoRA at equal batch size


def estimate_training_time(req: TrainingEstimateRequest) -> dict:
    """Cheap heuristic estimate so the UI can show 'approx. 18 min' before a job is queued.

    Not a substitute for profiling — intentionally conservative and explained as such
    in the response so users don't mistake it for a guarantee.
    """
    size_b = KNOWN_MODEL_SIZES_B.get(req.base_model, 7.0)
    base_tok_s = _GPU_TOK_S_PER_B_PARAMS.get(req.gpu_name, 200.0) / max(size_b, 0.5)

    if not req.use_lora:
        base_tok_s *= _FULL_FT_PENALTY
    if not req.load_in_4bit and req.use_lora:
        base_tok_s *= 0.85  # bf16 LoRA slightly slower than 4-bit LoRA on memory-bound steps

    effective_batch = req.per_device_batch_size * req.gradient_accumulation_steps
    total_tokens = req.dataset_total_tokens * req.epochs
    seconds = total_tokens / max(base_tok_s, 1.0)

    total_steps = max(1, total_tokens // (effective_batch * req.max_seq_len))

    return {
        "estimated_seconds": round(seconds),
        "estimated_minutes": round(seconds / 60, 1),
        "estimated_total_steps": int(total_steps),
        "assumed_throughput_tokens_per_sec": round(base_tok_s, 1),
        "model_size_b_params": size_b,
        "note": (
            "Heuristic estimate based on public QLoRA throughput benchmarks for the "
            f"selected GPU ({req.gpu_name}), not a profiled measurement of this exact job. "
            "Actual time varies with dataset shape, sequence packing, and disk/network I/O."
        ),
    }
