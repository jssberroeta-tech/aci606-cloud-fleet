"""Cloud Fleet Challenge - API FastAPI (persistencia SQL + ML + métricas + dashboard)."""
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from . import database
from .database import get_db
from .metrics import metrics
from .ml import ModelNotLoaded, model_service
from .models import Prediction, Telemetry
from .schemas import PredictIn, PredictOut, TelemetryIn, TelemetryOut

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fleet.api")

BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_FILE = BASE_DIR / "templates" / "dashboard.html"

TEL_FIELDS = ["timestamp", "x", "y", "speed", "battery", "motor_temp", "vibration", "motor_current", "obstacle_distance"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        database.init_db()
    except Exception as exc:  # la API arranca igual; /health informará db=down
        log.error("No se pudo inicializar la BD: %s", exc)
    model_service.load()
    yield


app = FastAPI(title="ACI606 Cloud Fleet Challenge", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------- middleware / errores
@app.middleware("http")
async def observe(request: Request, call_next):
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        ms = (time.perf_counter() - t0) * 1000.0
        route = request.scope.get("route")
        path = getattr(route, "path", None) or request.url.path
        if route is None:  # rutas desconocidas: no inflar cardinalidad
            path = "unmatched"
        metrics.record(request.method, path, status, ms)


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError):
    # Se omite "input": puede contener NaN/Inf y romper la serialización JSON (-> 500).
    errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(SQLAlchemyError)
async def db_handler(_: Request, exc: SQLAlchemyError):
    log.error("Error de base de datos: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "database unavailable"})


@app.exception_handler(Exception)
async def unhandled_handler(_: Request, exc: Exception):
    log.exception("Error no controlado: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "internal error"})


# ---------------------------------------------------------------- rutas
@app.get("/health")
def health():
    db_ok = True
    try:
        with database.engine.connect() as conn:  # despierta Neon si estaba suspendida
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        db_ok = False
        log.warning("health: BD no disponible: %s", exc)
    return {
        "status": "ok",
        "db": "up" if db_ok else "down",
        "db_backend": "sqlite" if database.IS_SQLITE else "postgresql",
        "model_loaded": model_service.loaded,
        "model_version": model_service.version,
        "uptime_seconds": round(time.time() - metrics.started_at, 1),
    }


@app.post("/telemetry", response_model=TelemetryOut)
def post_telemetry(sample: TelemetryIn, db: Session = Depends(get_db)):
    data = sample.model_dump()
    data["timestamp"] = sample.resolved_timestamp()
    row = Telemetry(**data)
    db.add(row)
    db.flush()  # obtiene row.id

    pred = None
    if model_service.loaded:
        try:
            pred = model_service.predict(data)
            metrics.record_inference(pred["inference_ms"])
            db.add(Prediction(telemetry_id=row.id, robot_id=row.robot_id, **pred))
        except Exception as exc:  # la ingesta nunca falla por culpa del modelo
            log.warning("predicción en ingesta omitida: %s", exc)
    db.commit()
    return TelemetryOut(
        id=row.id,
        robot_id=row.robot_id,
        predicted_status=pred["predicted_status"] if pred else None,
        confidence=pred["confidence"] if pred else None,
    )


def _row_dict(t: Telemetry, p: Prediction | None) -> dict:
    d = {"id": t.id, "robot_id": t.robot_id}
    d.update({f: getattr(t, f) for f in TEL_FIELDS})
    d["received_at"] = t.received_at
    d["predicted_status"] = p.predicted_status if p else None
    d["confidence"] = p.confidence if p else None
    return d


@app.get("/robots")
def list_robots(db: Session = Depends(get_db)):
    latest = select(func.max(Telemetry.id).label("mid")).group_by(Telemetry.robot_id).subquery()
    q = (
        select(Telemetry, Prediction)
        .join(latest, Telemetry.id == latest.c.mid)
        .outerjoin(Prediction, Prediction.telemetry_id == Telemetry.id)
        .order_by(Telemetry.robot_id)
    )
    counts = dict(db.execute(select(Telemetry.robot_id, func.count(Telemetry.id)).group_by(Telemetry.robot_id)).all())
    out = []
    for t, p in db.execute(q).all():
        d = _row_dict(t, p)
        d["samples"] = counts.get(t.robot_id, 0)
        out.append(d)
    return out


@app.get("/robots/{robot_id}/history")
def robot_history(
    robot_id: str,
    limit: int | None = Query(default=None, ge=1, le=1_000_000, description="Opcional: últimas N muestras"),
    db: Session = Depends(get_db),
):
    base = (
        select(Telemetry, Prediction)
        .outerjoin(Prediction, Prediction.telemetry_id == Telemetry.id)
        .where(Telemetry.robot_id == robot_id)
    )
    if limit:
        rows = db.execute(base.order_by(Telemetry.id.desc()).limit(limit)).all()[::-1]
    else:
        rows = db.execute(base.order_by(Telemetry.id.asc())).all()  # historial completo, sin límite fijo
    if not rows and not db.scalar(select(func.count(Telemetry.id)).where(Telemetry.robot_id == robot_id)):
        raise HTTPException(status_code=404, detail=f"robot '{robot_id}' sin telemetría")
    return [_row_dict(t, p) for t, p in rows]


@app.post("/predict", response_model=PredictOut)
def predict(sample: PredictIn):
    try:
        pred = model_service.predict(sample.model_dump())
    except ModelNotLoaded:
        raise HTTPException(status_code=503, detail="model not loaded")
    metrics.record_inference(pred["inference_ms"])
    return PredictOut(robot_id=sample.robot_id, **pred)


@app.get("/metrics")
def get_metrics(db: Session = Depends(get_db)):
    extra = {
        "model": {"loaded": model_service.loaded, "version": model_service.version},
        "db_backend": "sqlite" if database.IS_SQLITE else "postgresql",
        "requests_received": metrics.telemetry_requests,  # compatibilidad con el starter
    }
    try:
        extra["robots_connected"] = db.scalar(select(func.count(func.distinct(Telemetry.robot_id)))) or 0
        extra["telemetry_rows"] = db.scalar(select(func.count(Telemetry.id))) or 0
    except SQLAlchemyError:
        extra["robots_connected"] = extra["telemetry_rows"] = None
    return metrics.snapshot(extra)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(DASHBOARD_FILE.read_text(encoding="utf-8"))


@app.get("/")
def root():
    return {"service": "aci606-cloud-fleet-challenge", "docs": "/docs", "dashboard": "/dashboard", "health": "/health"}
