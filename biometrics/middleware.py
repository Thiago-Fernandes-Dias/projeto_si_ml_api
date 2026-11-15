from __future__ import annotations

import logging
import time
from collections.abc import Callable

from django.http import HttpRequest, HttpResponse

from biometrics.logging_utils import (
    correlation_context,
    infer_flow_from_path,
    log_structured,
)

logger = logging.getLogger("biometrics.http")

CORRELATION_HEADER = "X-Correlation-ID"


class CorrelationLoggingMiddleware:
    """Bind request correlation IDs and emit structured HTTP debug logs."""

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        raw_correlation = request.headers.get(CORRELATION_HEADER)
        flow = infer_flow_from_path(request.path)
        started = time.perf_counter()

        with correlation_context(raw_correlation, flow=flow) as correlation_id:
            log_structured(
                logger,
                logging.DEBUG,
                "ml_api.http.request_started",
                "Requisição HTTP recebida na ML API.",
                flow=flow,
                method=request.method,
                path=request.path,
                contentLength=request.META.get("CONTENT_LENGTH") or 0,
            )
            response = self.get_response(request)
            response[CORRELATION_HEADER] = correlation_id
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            log_structured(
                logger,
                logging.DEBUG,
                "ml_api.http.request_completed",
                "Requisição HTTP concluída na ML API.",
                flow=flow,
                method=request.method,
                path=request.path,
                statusCode=response.status_code,
                durationMs=elapsed_ms,
            )
            return response
