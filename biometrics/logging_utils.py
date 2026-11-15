from __future__ import annotations

import contextvars
import json
import logging
import re
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

_CORRELATION_ID_VAR: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "biometrics_correlation_id",
    default=None,
)

_FLOW_VAR: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "biometrics_flow",
    default=None,
)

_SAFE_CORRELATION_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")

_SENSITIVE_EXACT_KEYS = frozenset({
    "password",
    "targetpassword",
    "target_password",
    "confirmpassword",
    "confirm_password",
    "passwordattempt",
    "password_attempt",
    "senha",
    "senhahash",
    "senha_hash",
    "secret",
    "token",
    "accesstoken",
    "access_token",
    "authorization",
    "rawtimings",
    "raw_timings",
    "textentered",
    "text_entered",
    "features",
})

_SENSITIVE_SUBSTRINGS = (
    "secret",
    "authorization",
    "rawtimings",
    "raw_timings",
    "textentered",
    "text_entered",
    "senhahash",
    "senha_hash",
)


def _is_sensitive_key(raw_key: str) -> bool:
    lowered = raw_key.lower()
    if lowered in _SENSITIVE_EXACT_KEYS:
        return True
    if any(token in lowered for token in _SENSITIVE_SUBSTRINGS):
        return True
    if "password" in lowered and not lowered.endswith("length"):
        return True
    safe_token_flags = {"hastoken", "has_token", "tokenissued", "token_issued"}
    return "token" in lowered and lowered not in safe_token_flags

_STANDARD_RECORD_ATTRS = frozenset(logging.makeLogRecord({}).__dict__.keys())
_TRUE_STRINGS = frozenset({"1", "true", "yes", "on"})
_FALSE_STRINGS = frozenset({"0", "false", "no", "off"})


def resolve_debug_logs_enabled(
    *,
    default: bool,
    environ_map: Mapping[str, str],
) -> bool:
    """Resolve BIOMETRICS_DEBUG_LOGS -> DEBUG_LOGS -> default safely ignoring blank strings."""
    for env_key in ("BIOMETRICS_DEBUG_LOGS", "DEBUG_LOGS"):
        raw = environ_map.get(env_key)
        if raw is None:
            continue
        cleaned = raw.strip().lower()
        if not cleaned:
            continue
        if cleaned in _TRUE_STRINGS:
            return True
        if cleaned in _FALSE_STRINGS:
            return False
    return bool(default)


def sanitize_correlation_id(raw_value: str | None) -> str:
    """Validate incoming correlation IDs or mint a fresh identifier."""
    candidate = (raw_value or "").strip()
    if candidate and _SAFE_CORRELATION_PATTERN.match(candidate):
        return candidate
    return uuid.uuid4().hex


def get_correlation_id() -> str | None:
    """Return the active correlation ID for the current execution context."""
    return _CORRELATION_ID_VAR.get()


def get_active_flow() -> str | None:
    """Return the active business flow label for the current execution context."""
    return _FLOW_VAR.get()


@contextmanager
def correlation_context(
    correlation_id: str | None = None,
    *,
    flow: str | None = None,
) -> Iterator[str]:
    """Bind a correlation ID and optional flow name for downstream log records."""
    resolved_id = sanitize_correlation_id(correlation_id)
    corr_token = _CORRELATION_ID_VAR.set(resolved_id)
    flow_token = _FLOW_VAR.set(flow) if flow is not None else None
    try:
        yield resolved_id
    finally:
        if flow_token is not None:
            _FLOW_VAR.reset(flow_token)
        _CORRELATION_ID_VAR.reset(corr_token)


def infer_flow_from_path(path: str) -> str:
    """Map an HTTP route path to a stable domain flow name."""
    lowered = (path or "").lower()
    if "enrollment" in lowered:
        return "enrollment"
    if "authenticate" in lowered:
        return "login"
    if "health" in lowered:
        return "health"
    return "ml_api"


def sanitize_log_data(data: Mapping[str, Any]) -> dict[str, Any]:
    """Redact credentials, tokens, and nominal feature maps before emitting logs."""
    cleaned: dict[str, Any] = {}
    for key, value in data.items():
        if _is_sensitive_key(str(key)):
            cleaned[str(key)] = "[REDACTED]"
            continue
        if isinstance(value, Mapping):
            cleaned[str(key)] = sanitize_log_data(value)
        else:
            cleaned[str(key)] = value
    return cleaned


class StructuredJsonFormatter(logging.Formatter):
    """Render Python LogRecord instances as single-line JSON entries for stdout."""

    def __init__(self, service: str = "projeto_si_ml_api") -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=UTC).isoformat()
        payload: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "message": record.getMessage(),
        }

        correlation_id = getattr(record, "correlationId", None) or get_correlation_id()
        if correlation_id:
            payload["correlationId"] = correlation_id

        flow = getattr(record, "flow", None) or get_active_flow()
        if flow:
            payload["flow"] = flow

        event = getattr(record, "event", None)
        if event:
            payload["event"] = event

        extra_data: dict[str, Any] = {}
        record_data = getattr(record, "data", None)
        if isinstance(record_data, Mapping):
            extra_data.update(record_data)

        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_ATTRS or key in {"correlationId", "flow", "event", "data"}:
                continue
            extra_data[key] = value

        if extra_data:
            payload["data"] = sanitize_log_data(extra_data)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def log_structured(
    target_logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    *,
    flow: str | None = None,
    **fields: Any,
) -> None:
    """Emit a structured log record while skipping work when level is disabled."""
    if not target_logger.isEnabledFor(level):
        return
    extra: dict[str, Any] = {"event": event}
    if flow is not None:
        extra["flow"] = flow
    if fields:
        extra["data"] = sanitize_log_data(fields)
    target_logger.log(level, message, extra=extra)
