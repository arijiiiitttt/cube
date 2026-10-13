"""
HF Trainer-based fine-tuning core for Cube.

Replaces the previous from-scratch GPT/LLaMA implementation
(trainer/model_builder.py) with real Hugging Face causal LMs fine-tuned via
QLoRA. This is what actually moves the needle on speed/VRAM, because it lets
us fit larger effective batch sizes and longer sequences in the same GPU
memory budget, and use kernels (Flash Attention 2, fused optimizers) that a
hand-rolled attention implementation can't access.

Falls back gracefully (logged, not silent) when a requested feature isn't
available on the current hardware — e.g. Flash Attention 2 needs an Ampere+
GPU and the package installed; bf16 needs Ampere+; 4-bit needs a CUDA GPU.
"""
import os
import json
import logging
import time
from pathlib import Path

import redis
import torch

logger = logging.getLogger(__name__)
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")


def _bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    except ImportError:
        return False


class RedisProgressCallback:
    """HF TrainerCallback that streams step/loss/lr to Redis (pub/sub + a list for replay)
    and updates the Postgres row, so the frontend dashboard keeps working unmodified."""

    def __init__(self, job_id: str, redis_client, db_update_fn, log_every: int = 5):
        self.job_id = job_id
        self.redis = redis_client
        self.db_update = db_update_fn
        self.log_every = log_every
        self.t_start = time.time()

    def log(self, msg: str):
        logger.info(msg)
        self.redis.rpush(f"logs:{self.job_id}", msg)
        self.redis.expire(f"logs:{self.job_id}", 86400)

    def should_stop(self) -> bool:
        return bool(self.redis.get(f"stop:{self.job_id}"))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        step = state.global_step
        loss = logs["loss"]
        lr = logs.get("learning_rate", 0.0)
        epoch = logs.get("epoch", 0.0)

        self.redis.publish(
            f"training:{self.job_id}",
            json.dumps({"step": step, "train_loss": loss, "lr": lr, "epoch": epoch}),
        )
        if step % self.log_every == 0 or step == 1:
            elapsed = time.time() - self.t_start
            self.log(f"[STEP {step:05d}/{state.max_steps}] loss={loss:.4f} lr={lr:.2e} elapsed={elapsed:.0f}s")
            self.db_update(
                current_step=step,
                current_epoch=int(epoch),
                train_loss=round(loss, 6),
                lr=round(lr, 8),
            )
        if self.should_stop():
            control.should_training_stop = True

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        val_loss = metrics.get("eval_loss")
        if val_loss is not None:
            self.log(f"[EVAL] step={state.global_step} val_loss={val_loss:.4f}")
            self.db_update(val_loss=round(val_loss, 6))


class Trainer:
    """Thin orchestration layer around `transformers.Trainer` + PEFT.

    Public surface (job_id, config, train()) is unchanged from the previous
    implementation so celery_tasks.py barely needs to change.
    """

    def __init__(self, job_id: str, config: dict):
        self.job_id = job_id
        self.config = config
        self.redis = redis.from_url(REDIS_URL, decode_responses=True)
        self._db_conn = None

    # ── DB helpers (single persistent connection) ──────────
    def _get_conn(self):
        import psycopg2
        if self._db_conn is None or self._db_conn.closed:
            url = os.getenv("DATABASE_URL", "").replace("+asyncpg", "")
            self._db_conn = psycopg2.connect(url)
            self._db_conn.autocommit = False
        return self._db_conn

    def update_job_db(self, **kwargs):
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            sets = [f"{k}=%s" for k in kwargs]
            vals = list(kwargs.values()) + [self.job_id]
            cur.execute(f"UPDATE training_jobs SET {', '.join(sets)}, updated_at=NOW() WHERE id=%s", vals)
            conn.commit()
        except Exception as e:
            logger.warning(f"DB update failed: {e}")
            try:
                if self._db_conn:
                    self._db_conn.rollback()
            except Exception:
                self._db_conn = None

    def log(self, msg: str):
        logger.info(msg)
        self.redis.rpush(f"logs:{self.job_id}", msg)
        self.redis.expire(f"logs:{self.job_id}", 86400)

    # ── Model + tokenizer loading ───────────────────────────
    def _load_model_and_tokenizer(self, base_model: str):
        cfg = self.config
        use_4bit = cfg.get("load_in_4bit", True) and torch.cuda.is_available()
        use_flash_attn = cfg.get("use_flash_attention_2", True) and _flash_attn_available()
        compute_dtype = torch.bfloat16 if cfg.get("bnb_compute_dtype", "bf16") == "bf16" else torch.float16

        if not torch.cuda.is_available():
            self.log("[WARN] No CUDA GPU detected — falling back to CPU. This will be very slow; "
                      "QLoRA/4-bit/Flash-Attention are GPU-only and will be skipped.")
            use_4bit = False
            use_flash_attn = False

        if cfg.get("use_flash_attention_2", True) and not use_flash_attn:
            self.log("[WARN] Flash Attention 2 unavailable on this hardware/package set — "
                      "falling back to PyTorch SDPA attention.")

        if cfg.get("use_unsloth"):
            return self._load_with_unsloth(base_model, use_4bit, compute_dtype)

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quant_config = None
        if use_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=cfg.get("bnb_double_quant", True),
                bnb_4bit_compute_dtype=compute_dtype,
            )

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant_config,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
            device_map="auto" if torch.cuda.is_available() else None,
        )

        if cfg.get("use_lora", True):
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            if use_4bit:
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=cfg.get("use_gradient_checkpointing", True)
                )
            elif cfg.get("use_gradient_checkpointing", True):
                model.gradient_checkpointing_enable()

            target_modules = cfg.get("lora_target_modules") or self._guess_target_modules(model)
            lora_config = LoraConfig(
                r=cfg.get("lora_r", 16),
                lora_alpha=cfg.get("lora_alpha", 32),
                lora_dropout=cfg.get("lora_dropout", 0.05),
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            trainable, total = model.get_nb_trainable_parameters()
            self.log(f"[INFO] LoRA attached — trainable params: {trainable:,} / {total:,} "
                      f"({100 * trainable / total:.2f}%)")
        elif cfg.get("use_gradient_checkpointing", True):
            model.gradient_checkpointing_enable()

        return model, tokenizer

    def _load_with_unsloth(self, base_model: str, use_4bit: bool, compute_dtype):
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            self.log("[WARN] use_unsloth=True but unsloth isn't installed — falling back to standard HF+PEFT path.")
            self.config["use_unsloth"] = False
            return self._load_model_and_tokenizer(base_model)

        cfg = self.config
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=cfg.get("max_seq_len", 1024),
            dtype=compute_dtype,
            load_in_4bit=use_4bit,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=cfg.get("lora_target_modules") or [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
            ],
            use_gradient_checkpointing=cfg.get("use_gradient_checkpointing", True),
            bias="none",
        )
        self.log("[INFO] Loaded via Unsloth fast kernels")
        return model, tokenizer

    @staticmethod
    def _guess_target_modules(model) -> list[str]:
        # Mirrors app/schemas/training.py:TARGET_MODULE_PRESETS in the backend.
        # Duplicated (not imported) on purpose — training-service and backend
        # are separate containers/packages and don't share a Python path.
        presets = {
            "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "gemma2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        }
        model_type = getattr(model.config, "model_type", "")
        return presets.get(model_type, ["q_proj", "k_proj", "v_proj", "o_proj"])

    # ── Dataset prep (tokenize + pack) ───────────────────────
    def _build_dataset(self, data_dir: str, tokenizer, max_seq_len: int, pack: bool):
        """Loads a JSONL of {"text": ...} examples produced by data_processor.py,
        tokenizes with caching, and optionally packs short sequences together to
        eliminate per-example padding waste (a real throughput win, not cosmetic)."""
        from datasets import load_dataset

        ds = load_dataset(
            "json",
            data_files={
                "train": os.path.join(data_dir, "train.jsonl"),
                "validation": os.path.join(data_dir, "val.jsonl"),
            },
        )

        def tokenize_fn(batch):
            return tokenizer(batch["text"], truncation=True, max_length=max_seq_len)

        # `datasets` caches tokenization results on disk keyed by a hash of the
        # function + inputs, so re-running the same job (or a job with the same
        # dataset_id) skips re-tokenization entirely.
        ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"], desc="Tokenizing")

        if pack:
            ds = self._pack_sequences(ds, max_seq_len)

        return ds["train"], ds["validation"]

    @staticmethod
    def _pack_sequences(ds, max_seq_len: int):
        """Concatenate tokenized examples end-to-end and chunk into fixed-size
        blocks, so every training step is a full max_seq_len block with zero
        padding — this is the single biggest preprocessing-side speed win for
        datasets made of many short examples."""

        def pack_fn(batch):
            concatenated = sum(batch["input_ids"], [])
            n_chunks = len(concatenated) // max_seq_len
            chunks = [concatenated[i * max_seq_len:(i + 1) * max_seq_len] for i in range(n_chunks)]
            return {"input_ids": chunks, "attention_mask": [[1] * max_seq_len for _ in chunks]}

        return ds.map(pack_fn, batched=True, remove_columns=ds["train"].column_names, desc="Packing sequences")

    # ── Main entry point ─────────────────────────────────────
    def train(self, data_dir: str, output_dir: str) -> str:
        cfg = self.config
        base_model = cfg.get("base_model", "meta-llama/Llama-3.2-1B-Instruct")

        self.log(f"[INFO] CUDA available: {torch.cuda.is_available()} | Base model: {base_model}")
        self.update_job_db(status="running")

        model, tokenizer = self._load_model_and_tokenizer(base_model)

        if cfg.get("use_torch_compile", True) and torch.cuda.is_available():
            try:
                model = torch.compile(model)
                self.log("[INFO] torch.compile() enabled (first step will include compile warmup)")
            except Exception as e:
                self.log(f"[WARN] torch.compile failed, continuing uncompiled: {e}")

        train_ds, val_ds = self._build_dataset(
            data_dir, tokenizer, cfg.get("max_seq_len", 1024), cfg.get("pack_sequences", True)
        )
        self.log(f"[INFO] Train examples: {len(train_ds)} | Val examples: {len(val_ds)}")

        from transformers import (
            TrainingArguments,
            Trainer as HFTrainer,
            DataCollatorForLanguageModeling,
        )

        bf16 = cfg.get("mixed_precision", "bf16") == "bf16" and _bf16_supported()
        fp16 = cfg.get("mixed_precision", "bf16") == "fp16" and not bf16 and torch.cuda.is_available()
        if cfg.get("mixed_precision", "bf16") == "bf16" and not bf16:
            self.log("[WARN] bf16 requested but unsupported on this GPU — falling back to fp16.")
            fp16 = torch.cuda.is_available()

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=cfg.get("epochs", 3),
            per_device_train_batch_size=cfg.get("per_device_batch_size", 4),
            per_device_eval_batch_size=cfg.get("per_device_batch_size", 4),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
            learning_rate=cfg.get("learning_rate", 2e-4),
            warmup_ratio=cfg.get("warmup_ratio", 0.03),
            max_grad_norm=cfg.get("grad_clip", 1.0),
            lr_scheduler_type="cosine",
            optim=cfg.get("optimizer", "paged_adamw_8bit"),
            bf16=bf16,
            fp16=fp16,
            gradient_checkpointing=cfg.get("use_gradient_checkpointing", True),
            dataloader_num_workers=cfg.get("dataloader_num_workers", 2),
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            report_to=[],  # we push to MLflow ourselves below for full control over run naming
        )

        progress_cb = RedisProgressCallback(self.job_id, self.redis, self.update_job_db)
        total_steps = (
            len(train_ds) // (cfg.get("per_device_batch_size", 4) * cfg.get("gradient_accumulation_steps", 4))
        ) * cfg.get("epochs", 3)
        self.update_job_db(total_steps=max(1, total_steps))

        use_mlflow = self._start_mlflow(cfg, base_model)

        hf_trainer = HFTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=[progress_cb],
        )

        try:
            hf_trainer.train()
        except Exception as e:
            self.log(f"[ERROR] Training crashed: {e}")
            self.update_job_db(status="error", error_msg=str(e))
            if use_mlflow:
                self._end_mlflow(failed=True)
            raise

        final_path = str(Path(output_dir) / "final_adapter")
        hf_trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)

        self.log(f"[DONE] Training complete — adapter saved to {final_path}")
        self.update_job_db(status="done")
        if use_mlflow:
            self._log_mlflow_artifact(final_path)
            self._end_mlflow(failed=False)

        if self._db_conn and not self._db_conn.closed:
            self._db_conn.close()

        return final_path

    # ── MLflow (best-effort, never blocks training) ──────────
    def _start_mlflow(self, cfg: dict, base_model: str) -> bool:
        try:
            import mlflow
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
            mlflow.start_run(run_name=f"cube-{self.job_id[:8]}")
            mlflow.log_params({**cfg, "base_model": base_model})
            return True
        except Exception:
            self.log("[WARN] MLflow unavailable — training continues without tracking")
            return False

    def _log_mlflow_artifact(self, path: str):
        try:
            import mlflow
            mlflow.log_artifacts(path, artifact_path="adapter")
        except Exception:
            pass

    def _end_mlflow(self, failed: bool):
        try:
            import mlflow
            mlflow.end_run(status="FAILED" if failed else "FINISHED")
        except Exception:
            pass
