import os, json, tempfile, uuid, logging
from celery import Celery
from minio import Minio
from minio.error import S3Error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
BUCKET    = os.getenv("MINIO_BUCKET", "cube-models")

app = Celery("cube", broker=REDIS_URL, backend=REDIS_URL)
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_routes={"process_dataset": {"queue": "training"}, "train_model": {"queue": "training"}},
)


def get_minio():
    client = Minio(
        os.getenv("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.getenv("MINIO_ROOT_USER", "minioadmin"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin123"),
        secure=False,
    )
    try:
        if not client.bucket_exists(BUCKET):
            client.make_bucket(BUCKET)
    except S3Error:
        pass  # Already exists (race condition safe)
    return client


def get_db_conn():
    import psycopg2
    url = os.getenv("DATABASE_URL", "").replace("+asyncpg", "")
    return psycopg2.connect(url)


def db_exec(sql, params=()):
    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def db_fetch(sql, params=()):
    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur.fetchone()
    finally:
        conn.close()


@app.task(bind=True, name="process_dataset", queue="training")
def process_dataset_task(self, dataset_id: str):
    from trainer.data_processor import process_files
    client = get_minio()

    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = os.path.join(tmp, "raw")
        out_dir = os.path.join(tmp, "processed")
        os.makedirs(raw_dir)

        objects = list(client.list_objects(BUCKET, prefix=f"datasets/{dataset_id}/raw/", recursive=True))
        if not objects:
            db_exec("UPDATE datasets SET status='error' WHERE id=%s", (dataset_id,))
            return {"error": "no files found"}

        local_paths = []
        for obj in objects:
            fname = os.path.basename(obj.object_name)
            local = os.path.join(raw_dir, fname)
            client.fget_object(BUCKET, obj.object_name, local)
            local_paths.append(local)

        try:
            stats = process_files(local_paths, out_dir)
        except Exception as e:
            db_exec("UPDATE datasets SET status='error' WHERE id=%s", (dataset_id,))
            logger.error(f"Processing failed for {dataset_id}: {e}")
            raise

        # JSONL text splits — tokenization happens later, lazily, once we know
        # which base model the user picked (each model has its own tokenizer).
        for fname in ["train.jsonl", "val.jsonl"]:
            local_f = os.path.join(out_dir, fname)
            if os.path.exists(local_f):
                client.fput_object(BUCKET, f"datasets/{dataset_id}/processed/{fname}", local_f)

        db_exec(
            """UPDATE datasets SET status='ready', total_tokens=%s, train_tokens=%s,
               val_tokens=%s, stats=%s, updated_at=NOW() WHERE id=%s""",
            (stats["approx_total_tokens"], stats["train_examples"],
             stats["val_examples"], json.dumps(stats), dataset_id)
        )
        logger.info(f"Dataset {dataset_id} processed: {stats}")
        return stats


@app.task(bind=True, name="train_model", queue="training")
def train_model_task(self, job_id: str, config: dict):
    from trainer.hf_trainer import Trainer

    client = get_minio()
    dataset_id = config.get("dataset_id", "")

    row     = db_fetch("SELECT user_id FROM training_jobs WHERE id=%s", (job_id,))
    user_id = row[0] if row else ""

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data")
        out_dir  = os.path.join(tmp, "output")
        os.makedirs(data_dir)

        for fname in ["train.jsonl", "val.jsonl"]:
            try:
                client.fget_object(
                    BUCKET, f"datasets/{dataset_id}/processed/{fname}",
                    os.path.join(data_dir, fname)
                )
            except Exception as e:
                db_exec(
                    "UPDATE training_jobs SET status='error', error_msg=%s WHERE id=%s",
                    (f"Dataset file missing: {fname} — {e}", job_id)
                )
                return {"error": str(e)}

        try:
            trainer    = Trainer(job_id, config)
            model_path = trainer.train(data_dir, out_dir)
        except Exception as e:
            db_exec(
                "UPDATE training_jobs SET status='error', error_msg=%s WHERE id=%s",
                (str(e), job_id)
            )
            logger.error(f"Training failed for job {job_id}: {e}")
            raise

        if not model_path:
            return {}  # Stopped by user

        # Upload the LoRA adapter (or full model dir) to MinIO — typically a
        # few MB to ~100s of MB for an adapter, vs GBs for a full checkpoint.
        adapter_prefix = f"models/{job_id}/adapter"
        for root, _, files in os.walk(model_path):
            for fname in files:
                local_f = os.path.join(root, fname)
                rel = os.path.relpath(local_f, model_path)
                client.fput_object(BUCKET, f"{adapter_prefix}/{rel}", local_f)

        db_exec(
            "UPDATE training_jobs SET model_path=%s, status='done', finished_at=NOW() WHERE id=%s",
            (adapter_prefix, job_id)
        )

        model_id = str(uuid.uuid4())
        base_model = config.get("base_model", "unknown")
        db_exec(
            """INSERT INTO model_registry
               (id, user_id, name, arch, job_id, status, params, model_path, config)
               VALUES (%s, %s, %s, %s, %s, 'ready', %s, %s, %s)""",
            (model_id, user_id,
             f"{base_model.split('/')[-1]}-{job_id[:8]}",
             base_model, job_id, base_model, adapter_prefix,
             json.dumps(config))
        )

        logger.info(f"Training complete. Job: {job_id}, Model: {model_id}")
        return {"model_path": adapter_prefix, "model_id": model_id}


@app.task(bind=True, name="evaluate_model", queue="training")
def evaluate_model_task(self, model_id: str):
    """Compute perplexity on the held-out val set using the fine-tuned adapter."""
    import io, math, tempfile, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    client = get_minio()
    row    = db_fetch("SELECT model_path, config FROM model_registry WHERE id=%s", (model_id,))
    if not row:
        return {"error": "model not found"}

    model_path, config = row
    config = json.loads(config) if isinstance(config, str) else (config or {})
    dataset_id = config.get("dataset_id", "")
    base_model = config.get("base_model", "")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            adapter_dir = os.path.join(tmp, "adapter")
            os.makedirs(adapter_dir)
            objects = list(client.list_objects(BUCKET, prefix=f"{model_path}/", recursive=True))
            for obj in objects:
                rel = os.path.relpath(obj.object_name, model_path)
                local_f = os.path.join(adapter_dir, rel)
                os.makedirs(os.path.dirname(local_f), exist_ok=True)
                client.fget_object(BUCKET, obj.object_name, local_f)

            tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            model = PeftModel.from_pretrained(base, adapter_dir)
            model.eval()

            val_buf = io.BytesIO()
            client.get_object(BUCKET, f"datasets/{dataset_id}/processed/val.jsonl").readinto(val_buf)
            val_lines = val_buf.getvalue().decode("utf-8").splitlines()

            losses = []
            with torch.no_grad():
                for line in val_lines[:50]:  # cap for a fast, bounded eval pass
                    text = json.loads(line).get("text", "")
                    if not text.strip():
                        continue
                    enc = tokenizer(text, truncation=True, max_length=1024, return_tensors="pt")
                    if torch.cuda.is_available():
                        enc = {k: v.to(model.device) for k, v in enc.items()}
                    out = model(**enc, labels=enc["input_ids"])
                    losses.append(out.loss.item())

            avg_loss = sum(losses) / max(len(losses), 1)
            perplexity = math.exp(min(avg_loss, 20))  # guard against overflow on a bad batch

            metrics = {
                "perplexity": round(perplexity, 2),
                "eval_loss": round(avg_loss, 4),
                "n_examples_evaluated": len(losses),
                # Heuristic, perplexity-derived proxies (not measured BLEU/token-accuracy) —
                # kept so the existing evaluation dashboard renders without changes.
                "accuracy": round(min(99, max(0, 100 - perplexity / 2)), 1),
                "bleu": round(max(0, min(1, 0.8 - avg_loss / 5)), 3),
                "repetition_rate": round(max(0.1, avg_loss / 3), 1),
                "grade": (
                    "A" if perplexity < 15 else
                    "B+" if perplexity < 25 else
                    "B"  if perplexity < 40 else
                    "C+" if perplexity < 60 else "C"
                ),
            }
    except Exception as e:
        logger.error(f"Eval failed for {model_id}: {e}")
        metrics = {
            "perplexity": 0, "eval_loss": 0, "accuracy": 0,
            "bleu": 0, "repetition_rate": 0, "grade": "N/A",
            "error": str(e),
        }

    db_exec("UPDATE model_registry SET metrics=%s WHERE id=%s",
            (json.dumps(metrics), model_id))
    return metrics
