import io
import json
import logging

import pytest
from rest_framework.test import APIClient

from biometrics.logging_utils import (
    StructuredJsonFormatter,
    correlation_context,
    log_structured,
    resolve_debug_logs_enabled,
)
from tests.conftest import (
    CMU_SAMPLE_VECTOR,
    create_user_in_db,
    make_enrollment_payload,
)


def test_resolve_debug_logs_enabled_respects_production_and_compose_overrides() -> None:
    assert resolve_debug_logs_enabled(default=False, environ_map={}) is False
    assert resolve_debug_logs_enabled(default=True, environ_map={}) is True
    assert (
        resolve_debug_logs_enabled(default=False, environ_map={"DEBUG_LOGS": "true"})
        is True
    )
    assert (
        resolve_debug_logs_enabled(
            default=False,
            environ_map={"DEBUG_LOGS": "true", "BIOMETRICS_DEBUG_LOGS": "false"},
        )
        is False
    )
    assert (
        resolve_debug_logs_enabled(
            default=False,
            environ_map={"BIOMETRICS_DEBUG_LOGS": "   ", "DEBUG_LOGS": "1"},
        )
        is True
    )


def test_structured_json_formatter_redacts_sensitive_fields_and_keeps_correlation() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(StructuredJsonFormatter(service="projeto_si_ml_api"))

    test_logger = logging.getLogger("biometrics.test_formatter")
    original_level = test_logger.level
    original_propagate = test_logger.propagate
    test_logger.setLevel(logging.DEBUG)
    test_logger.propagate = False
    test_logger.addHandler(handler)
    try:
        with correlation_context("corr-abc-123", flow="login"):
            log_structured(
                test_logger,
                logging.DEBUG,
                "ml_api.test.event",
                "Evento de diagnóstico",
                username="ana",
                password="secret-password",
                passwordLength=10,
                tokenIssued=True,
                accessToken="jwt-token-value",
                features={"H.t.0": 0.12},
                vectorDimension=31,
            )
    finally:
        test_logger.removeHandler(handler)
        test_logger.setLevel(original_level)
        test_logger.propagate = original_propagate

    parsed = json.loads(stream.getvalue().strip())
    assert parsed["level"] == "DEBUG"
    assert parsed["service"] == "projeto_si_ml_api"
    assert parsed["correlationId"] == "corr-abc-123"
    assert parsed["flow"] == "login"
    assert parsed["event"] == "ml_api.test.event"
    assert parsed["data"]["username"] == "ana"
    assert parsed["data"]["passwordLength"] == 10
    assert parsed["data"]["tokenIssued"] is True
    assert parsed["data"]["vectorDimension"] == 31
    assert parsed["data"]["password"] == "[REDACTED]"
    assert parsed["data"]["accessToken"] == "[REDACTED]"
    assert parsed["data"]["features"] == "[REDACTED]"


@pytest.mark.django_db
def test_enrollment_and_authenticate_emit_correlated_debug_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    biometrics_logger = logging.getLogger("biometrics")
    original_level = biometrics_logger.level
    biometrics_logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="biometrics")

    try:
        client = APIClient()
        create_user_in_db(username="maria_log")

        enroll_payload = make_enrollment_payload(username="maria_log", count=5, seed=9)
        enroll_res = client.post(
            "/api/enrollment",
            enroll_payload,
            format="json",
            HTTP_X_CORRELATION_ID="corr-enroll-001",
        )
        assert enroll_res.status_code == 201
        assert enroll_res.headers["X-Correlation-ID"] == "corr-enroll-001"

        auth_res = client.post(
            "/api/authenticate",
            {"username": "maria_log", "vector": CMU_SAMPLE_VECTOR},
            format="json",
            HTTP_X_CORRELATION_ID="corr-login-002",
        )
        assert auth_res.status_code == 200
        assert auth_res.headers["X-Correlation-ID"] == "corr-login-002"
    finally:
        biometrics_logger.setLevel(original_level)

    debug_events = [
        getattr(rec, "event", None)
        for rec in caplog.records
        if rec.levelno == logging.DEBUG and getattr(rec, "event", None)
    ]
    assert "ml_api.http.request_started" in debug_events
    assert "ml_api.enrollment.training_started" in debug_events
    assert "ml.train.completed" in debug_events
    assert "ml_api.enrollment.succeeded" in debug_events
    assert "ml_api.login.inference_starting" in debug_events
    assert "ml.classify.completed" in debug_events
    assert "ml_api.login.responded" in debug_events


@pytest.mark.django_db
def test_production_info_level_suppresses_debug_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    biometrics_logger = logging.getLogger("biometrics")
    original_level = biometrics_logger.level
    biometrics_logger.setLevel(logging.INFO)
    caplog.set_level(logging.DEBUG)

    try:
        client = APIClient()
        create_user_in_db(username="maria_prod")

        enroll_res = client.post(
            "/api/enrollment",
            make_enrollment_payload(username="maria_prod", count=5, seed=11),
            format="json",
        )
        assert enroll_res.status_code == 201

        auth_res = client.post(
            "/api/authenticate",
            {"username": "maria_prod", "vector": CMU_SAMPLE_VECTOR},
            format="json",
        )
        assert auth_res.status_code == 200
    finally:
        biometrics_logger.setLevel(original_level)

    biometric_debug_records = [
        rec
        for rec in caplog.records
        if rec.name.startswith("biometrics") and rec.levelno < logging.INFO
    ]
    assert biometric_debug_records == []
