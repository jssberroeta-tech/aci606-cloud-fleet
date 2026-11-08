"""Tablas SQLAlchemy: Telemetry (historial completo) y Prediction."""
import time

from sqlalchemy import Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class Telemetry(Base):
    __tablename__ = "telemetry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    robot_id: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    speed: Mapped[float] = mapped_column(Float, nullable=False)
    battery: Mapped[float] = mapped_column(Float, nullable=False)
    motor_temp: Mapped[float] = mapped_column(Float, nullable=False)
    vibration: Mapped[float] = mapped_column(Float, nullable=False)
    motor_current: Mapped[float] = mapped_column(Float, nullable=False)
    obstacle_distance: Mapped[float] = mapped_column(Float, nullable=False)
    received_at: Mapped[float] = mapped_column(Float, default=time.time, nullable=False)

    __table_args__ = (Index("ix_telemetry_robot_id_id", "robot_id", "id"),)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    telemetry_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("telemetry.id", ondelete="CASCADE"), index=True, nullable=False
    )
    robot_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    predicted_status: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    inference_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[float] = mapped_column(Float, default=time.time, nullable=False)
