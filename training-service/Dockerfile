# CPU image — for local dev / demoing the UI flow without a GPU.
# QLoRA 4-bit (bitsandbytes) and Flash Attention 2 are CUDA-only, so on this
# image TrainingConfig.load_in_4bit / use_flash_attention_2 are auto-disabled
# at runtime (see trainer/hf_trainer.py) and training falls back to a slow
# fp32 CPU run — fine for testing the pipeline end-to-end on a tiny model,
# not for real fine-tuning. Use docker-compose.gpu.yml for that.
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["celery", "-A", "tasks.celery_tasks", "worker", "--loglevel=info", "--concurrency=1"]
