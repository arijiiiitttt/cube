from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings


def _build_engine():
    url = settings.DATABASE_URL
    # ssl=require is needed for NeonDB / hosted Postgres; skip for local dev
    connect_args = {}
    if "sslmode=require" in url or "ssl=require" in url:
        connect_args["ssl"] = "require"
    # Strip sslmode from URL since asyncpg doesn't parse it as a query param
    url = url.replace("?sslmode=require", "").replace("&sslmode=require", "")
    url = url.replace("?ssl=require", "").replace("&ssl=require", "")

    return create_async_engine(
        url,
        echo=False,
        future=True,
        pool_size=5,
        max_overflow=10,
        connect_args=connect_args,
    )


engine = _build_engine()

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    pass


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
