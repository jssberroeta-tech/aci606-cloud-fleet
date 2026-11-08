"""Esquemas Pydantic estrictos. Datos inválidos / NaN / Inf -> 422 (nunca 5xx)."""
import time
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

Num = dict(allow_inf_nan=False)  # rechaza NaN e Infinity


class TelemetryIn(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    robot_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.\-]+$")
    timestamp: Optional[float] = Field(default=None, ge=0, le=4_102_444_800, **Num)
    x: float = Field(ge=-100_000, le=100_000, **Num)
    y: float = Field(ge=-100_000, le=100_000, **Num)
    speed: float = Field(ge=0, le=100, **Num)
    battery: float = Field(ge=0, le=100, **Num)
    motor_temp: float = Field(ge=-60, le=300, **Num)
    vibration: float = Field(ge=0, le=1_000, **Num)
    motor_current: float = Field(ge=0, le=1_000, **Num)
    obstacle_distance: float = Field(ge=0, le=100_000, **Num)

    def resolved_timestamp(self) -> float:
        return self.timestamp if self.timestamp is not None else time.time()


class PredictIn(TelemetryIn):
    """Mismo contrato de entrada que /telemetry."""


class PredictOut(BaseModel):
    robot_id: str
    predicted_status: str
    confidence: float
    model_version: str
    inference_ms: float


class TelemetryOut(BaseModel):
    status: str = "ok"
    id: int
    robot_id: str
    predicted_status: Optional[str] = None
    confidence: Optional[float] = None
