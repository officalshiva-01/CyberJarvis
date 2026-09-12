import time
from contextlib import contextmanager
from typing import Generator, Optional, Dict, Any

from observability.logging import get_logger

logger = get_logger("CyberJarvis.Tracer")


class Tracer:
    """Simple in-memory and logging tracer for measuring execution performance."""

    def __init__(self):
        self.spans: list[Dict[str, Any]] = []

    def log_span(self, name: str, duration_ms: float, metadata: Optional[Dict[str, Any]] = None):
        span_data = {
            "name": name,
            "duration_ms": round(duration_ms, 2),
            "metadata": metadata or {},
        }
        self.spans.append(span_data)
        logger.info(f"Span '{name}' completed in {duration_ms:.2f}ms | metadata: {metadata}")


@contextmanager
def trace_span(name: str, tracer: Optional[Tracer] = None, metadata: Optional[Dict[str, Any]] = None) -> Generator[None, None, None]:
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        if tracer:
            tracer.log_span(name, duration_ms, metadata)
        else:
            logger.debug(f"Span '{name}' executed in {duration_ms:.2f}ms")
