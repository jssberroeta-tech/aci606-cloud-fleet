"""Carga de models/model.joblib e inferencia en vivo."""
import json
import logging
import os
import threading
import time
from typing import Optional

import joblib
import pandas as pd

log = logging.getLogger("fleet.ml")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
DEFAULT_FEATURES = ["speed", "battery", "motor_temp", "vibration", "motor_current", "obstacle_distance"]


class ModelNotLoaded(RuntimeError):
    pass


class ModelService:
    def __init__(self) -> None:
        self.pipeline = None
        self.features = list(DEFAULT_FEATURES)
        self.version = os.environ.get("MODEL_VERSION", "v1")
        self.meta: dict = {}
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    def load(self, path: Optional[str] = None) -> bool:
        path = path or MODEL_PATH
        if not os.path.exists(path):
            log.error("Modelo no encontrado en %s", path)
            return False
        with self._lock:
            self.pipeline = joblib.load(path)
            meta_path = os.path.join(os.path.dirname(path) or ".", "model_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as fh:
                    self.meta = json.load(fh)
                self.features = self.meta.get("features", self.features)
                self.version = self.meta.get("model_version", self.version)
            self._warmup()
        log.info("Modelo %s cargado (%d features)", self.version, len(self.features))
        return True

    def _warmup(self) -> None:
        try:
            self.predict({f: 1.0 for f in self.features})
        except Exception as exc:  # pragma: no cover
            log.warning("warm-up falló: %s", exc)

    def predict(self, sample: dict) -> dict:
        if not self.loaded:
            raise ModelNotLoaded("model.joblib no está cargado")
        t0 = time.perf_counter()
        row = pd.DataFrame([[float(sample[f]) for f in self.features]], columns=self.features)
        proba = self.pipeline.predict_proba(row)[0]
        idx = int(proba.argmax())
        label = str(self.pipeline.classes_[idx])
        return {
            "predicted_status": label,
            "confidence": round(float(proba[idx]), 4),
            "model_version": self.version,
            "inference_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }


model_service = ModelService()
