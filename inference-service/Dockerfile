FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic minio \
    transformers==4.45.2 peft==0.13.0 accelerate==0.34.2 bitsandbytes==0.44.1
COPY inference-service/server.py .
EXPOSE 8001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
