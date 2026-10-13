from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.config import settings
from app.core.database import init_db
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Import all models so SQLAlchemy registers them before create_all
    import app.models.dataset   # noqa: F401
    import app.models.model     # noqa: F401
    import app.models.job       # noqa: F401

    await init_db()
    logger.info("Database tables initialised")

    # Ensure MinIO bucket exists (non-fatal if MinIO not running yet)
    try:
        from minio import Minio
        client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ROOT_USER,
            secret_key=settings.MINIO_ROOT_PASSWORD,
            secure=settings.MINIO_SECURE,
        )
        if not client.bucket_exists(settings.MINIO_BUCKET):
            client.make_bucket(settings.MINIO_BUCKET)
            logger.info(f"Created MinIO bucket: {settings.MINIO_BUCKET}")
        else:
            logger.info(f"MinIO bucket '{settings.MINIO_BUCKET}' already exists")
    except Exception as e:
        logger.warning(f"MinIO init skipped: {e}")

    yield


app = FastAPI(
    title="Cube API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Middleware: request timing ────────────────────────────────────────────────
@app.middleware("http")
async def add_timing(request: Request, call_next):
    start    = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(round(time.time() - start, 4))
    return response


# ── Routes ────────────────────────────────────────────────────────────────────
from app.api.routes import upload, dataset, models, training, evaluation, deploy, inference  # noqa: E402

app.include_router(upload.router,     prefix="/upload",    tags=["upload"])
app.include_router(dataset.router,    prefix="/dataset",   tags=["dataset"])
app.include_router(models.router,     prefix="/models",    tags=["models"])
app.include_router(training.router,   prefix="/training",  tags=["training"])
app.include_router(evaluation.router, prefix="/eval",      tags=["evaluation"])
app.include_router(inference.router,  prefix="/inference", tags=["inference"])
app.include_router(deploy.router,     prefix="/deploy",    tags=["deploy"])


@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok", "service": "Cube API", "version": "0.1.0"}
