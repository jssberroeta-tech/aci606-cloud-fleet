"""Conexión SQLAlchemy: PostgreSQL (Neon) en la nube; SQLite local como fallback."""
import logging
import os
import threading
import time

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

log = logging.getLogger("fleet.db")


def normalize_database_url(url: str) -> str:
    """postgres:// | postgresql://  ->  postgresql+psycopg:// (driver psycopg 3)."""
    url = url.strip()
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


_raw_url = os.environ.get("DATABASE_URL", "").strip()
DATABASE_URL = normalize_database_url(_raw_url) if _raw_url else "sqlite:///./fleet.db"
IS_SQLITE = DATABASE_URL.startswith("sqlite")

_engine_kwargs = {"pool_pre_ping": True}  # reconecta si Neon suspendió la base
if IS_SQLITE:
    _engine_kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
else:
    _engine_kwargs.update(pool_recycle=240, pool_size=5, max_overflow=5)

engine = create_engine(DATABASE_URL, **_engine_kwargs)

if IS_SQLITE:

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


_ready = False
_lock = threading.Lock()


def init_db(retries: int = 4, delay: float = 2.0) -> None:
    """Crea las tablas (idempotente). Reintenta: Neon puede tardar en despertar."""
    global _ready
    with _lock:
        if _ready:
            return
        from . import models  # noqa: F401  (registra las tablas)

        last = None
        for attempt in range(1, retries + 1):
            try:
                Base.metadata.create_all(engine)
                _ready = True
                log.info("Base de datos lista (%s)", "sqlite" if IS_SQLITE else "postgresql")
                return
            except Exception as exc:  # pragma: no cover
                last = exc
                log.warning("init_db intento %d/%d falló: %s", attempt, retries, exc)
                time.sleep(delay)
        raise last


def get_db():
    if not _ready:
        init_db(retries=2, delay=1.0)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
