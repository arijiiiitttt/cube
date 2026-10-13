from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.security import get_current_user_id
from app.core.config import settings
from app.models.dataset import Dataset
from minio import Minio
from minio.error import S3Error
import uuid, io, logging

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED = {".txt", ".pdf", ".csv", ".py", ".js", ".ts", ".go", ".rs", ".md", ".json", ".cpp", ".c", ".java", ".rb"}


def get_minio() -> Minio:
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ROOT_USER,
        secret_key=settings.MINIO_ROOT_PASSWORD,
        secure=settings.MINIO_SECURE,
    )


@router.post("/")
async def upload_files(
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    if not files:
        raise HTTPException(400, "No files provided")
    if len(files) > 20:
        raise HTTPException(400, "Max 20 files per upload")

    client     = get_minio()
    dataset_id = str(uuid.uuid4())
    uploaded, total_bytes = [], 0

    for f in files:
        if not f.filename:
            continue
        ext = ("." + f.filename.rsplit(".", 1)[-1].lower()) if "." in f.filename else ""
        if ext not in ALLOWED:
            raise HTTPException(
                400,
                f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED))}",
            )

        content = await f.read()
        size    = len(content)

        if size == 0:
            raise HTTPException(400, f"{f.filename} is empty")
        if size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(413, f"{f.filename} exceeds {settings.MAX_FILE_SIZE_MB}MB limit")

        key = f"datasets/{dataset_id}/raw/{f.filename}"
        try:
            client.put_object(
                settings.MINIO_BUCKET,
                key,
                io.BytesIO(content),
                size,
                content_type=f.content_type or "application/octet-stream",
            )
        except S3Error as e:
            logger.error(f"MinIO upload failed: {e}")
            raise HTTPException(503, "Storage service unavailable. Make sure MinIO is running.")

        uploaded.append(key)
        total_bytes += size

    if not uploaded:
        raise HTTPException(400, "No valid files were uploaded")

    dataset = Dataset(
        id=dataset_id,
        user_id=user_id,
        name=f"Dataset-{dataset_id[:8]}",
        file_paths=uploaded,
        raw_bytes=total_bytes,
        status="pending",
    )
    db.add(dataset)
    await db.commit()

    logger.info(f"User {user_id} uploaded {len(uploaded)} files → dataset {dataset_id}")
    return {
        "dataset_id":    dataset_id,
        "files_uploaded": len(uploaded),
        "total_bytes":   total_bytes,
        "files":         [{"name": p.split("/")[-1], "path": p} for p in uploaded],
    }
