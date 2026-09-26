"""Structured logging for SAT-AI.

Two rendering modes, selected by ``SATAI_LOG_FORMAT``:

``console``
    Human-readable, for local development.
``json``
    One JSON object per line, for production log aggregation.

Long-running geospatial jobs are effectively invisible without structured logs
that carry the scene, tile, and run identifiers. A plain ``print`` in a loop
over 4,000 tiles tells you nothing about which tile failed.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, ClassVar

from satai.config import LogFormat, get_settings

_CONFIGURED = False

_RESERVED = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)


class JSONFormatter(logging.Formatter):
    """Render a log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Readable single-line output with any structured fields appended."""

    _COLOURS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    _RESET = "\033[0m"

    def __init__(self, *, colour: bool = True) -> None:
        super().__init__()
        self.colour = colour and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=UTC).strftime("%H:%M:%S")
        level = record.levelname
        if self.colour:
            level = f"{self._COLOURS.get(level, '')}{level:<8}{self._RESET}"
        else:
            level = f"{level:<8}"

        extras = {
            k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")
        }
        suffix = ""
        if extras:
            suffix = "  " + " ".join(f"{k}={v}" for k, v in extras.items())

        line = f"{ts} {level} {record.name:<28} {record.getMessage()}{suffix}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging(*, force: bool = False) -> None:
    """Install SAT-AI's log handler on the root logger. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    settings = get_settings()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JSONFormatter() if settings.log_format is LogFormat.JSON else ConsoleFormatter()
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level)

    # Third-party libraries are chatty; raise their floor.
    for noisy in ("urllib3", "botocore", "matplotlib", "rasterio", "fiona", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.

    Use structured fields rather than string interpolation::

        log.info("tile processed", extra={"tile_id": tid, "duration_s": dt})
    """
    configure_logging()
    return logging.getLogger(name)
