"""Observabilidad en memoria: solicitudes, errores, latencias P50/P95 y uptime."""
import threading
import time
from collections import Counter, deque

import numpy as np


class Metrics:
    def __init__(self, window: int = 5000) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.requests_total = 0
        self.errors_4xx = 0
        self.errors_5xx = 0
        self.by_route: Counter = Counter()
        self.by_status: Counter = Counter()
        self.telemetry_requests = 0
        self.predict_requests = 0
        self.predict_errors = 0
        self.predict_latency = deque(maxlen=window)  # ms, extremo a extremo en el servidor
        self.inference = deque(maxlen=window)  # ms, solo el modelo
        self.all_latency = deque(maxlen=window)

    def record(self, method: str, route: str, status: int, ms: float) -> None:
        with self._lock:
            self.requests_total += 1
            self.by_route[f"{method} {route}"] += 1
            self.by_status[str(status)] += 1
            self.all_latency.append(ms)
            if 400 <= status < 500:
                self.errors_4xx += 1
            elif status >= 500:
                self.errors_5xx += 1
            if route == "/telemetry" and method == "POST":
                self.telemetry_requests += 1
            if route == "/predict" and method == "POST":
                self.predict_requests += 1
                self.predict_latency.append(ms)
                if status >= 400:
                    self.predict_errors += 1

    def record_inference(self, ms: float) -> None:
        with self._lock:
            self.inference.append(ms)

    @staticmethod
    def _pct(values, q):
        return round(float(np.percentile(values, q)), 3) if values else None

    @staticmethod
    def _avg(values):
        return round(float(np.mean(values)), 3) if values else None

    def snapshot(self, extra: dict | None = None) -> dict:
        with self._lock:
            pl, inf, al = list(self.predict_latency), list(self.inference), list(self.all_latency)
            data = {
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "started_at": self.started_at,
                "requests_total": self.requests_total,
                "telemetry_requests": self.telemetry_requests,
                "predict_requests": self.predict_requests,
                "errors_total": self.errors_4xx + self.errors_5xx,
                "errors_4xx": self.errors_4xx,
                "errors_5xx": self.errors_5xx,
                "predict_errors": self.predict_errors,
                "avg_predict_latency_ms": self._avg(pl),
                "predict_latency_ms": {
                    "count": len(pl),
                    "avg": self._avg(pl),
                    "p50": self._pct(pl, 50),
                    "p95": self._pct(pl, 95),
                    "p99": self._pct(pl, 99),
                },
                "avg_inference_ms": self._avg(inf),
                "request_latency_ms": {"p50": self._pct(al, 50), "p95": self._pct(al, 95)},
                "requests_by_route": dict(self.by_route),
                "status_codes": dict(self.by_status),
            }
        if extra:
            data.update(extra)
        return data


metrics = Metrics()
