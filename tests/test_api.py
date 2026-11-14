import pytest
from rest_framework.test import APIClient

from biometrics.models import KeystrokeSample, SampleLabel, SampleSource
from tests.conftest import CMU_SAMPLE_FEATURES, make_section_5_3_payload


@pytest.mark.django_db
def test_health_endpoint() -> None:
    client = APIClient()
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


@pytest.mark.django_db
def test_full_flow_register_enrollment_and_authenticate() -> None:
    client = APIClient()

    reg_res = client.post(
        "/api/register",
        {"username": "thiago_dias", "password": ".tie5Roanl"},
        format="json",
    )
    assert reg_res.status_code == 201
    assert reg_res.json()["username"] == "thiago_dias"
    assert reg_res.json()["success"] is True

    dup_res = client.post(
        "/api/register",
        {"username": "thiago_dias", "password": ".tie5Roanl"},
        format="json",
    )
    assert dup_res.status_code == 409

    pre_auth = client.post(
        "/api/authenticate",
        {
            "username": "thiago_dias",
            "password": ".tie5Roanl",
            "features": CMU_SAMPLE_FEATURES,
        },
        format="json",
    )
    assert pre_auth.status_code == 409

    enroll_payload = make_section_5_3_payload(username="thiago_dias", count=5, seed=42)
    enroll_res = client.post("/api/enrollment", enroll_payload, format="json")
    assert enroll_res.status_code == 201
    enroll_data = enroll_res.json()
    assert enroll_data["username"] == "thiago_dias"
    assert enroll_data["modelVersion"] == 1
    assert enroll_data["featureCount"] == len(CMU_SAMPLE_FEATURES)
    assert enroll_data["samplesRegistered"] == 5

    dup_enroll = client.post("/api/enrollment", enroll_payload, format="json")
    assert dup_enroll.status_code == 409

    wrong_pw = client.post(
        "/api/authenticate",
        {
            "username": "thiago_dias",
            "password": "wrong-password",
            "features": CMU_SAMPLE_FEATURES,
        },
        format="json",
    )
    assert wrong_pw.status_code == 401
    assert wrong_pw.json()["decision"] == "INVALID_CREDENTIALS"
    assert KeystrokeSample.objects.filter(source=SampleSource.LOGIN).count() == 0

    unknown_user = client.post(
        "/api/authenticate",
        {
            "username": "ghost_user",
            "password": ".tie5Roanl",
            "features": CMU_SAMPLE_FEATURES,
        },
        format="json",
    )
    assert unknown_user.status_code == 401
    assert unknown_user.json()["decision"] == "INVALID_CREDENTIALS"

    genuine_res = client.post(
        "/api/authenticate",
        {
            "username": "thiago_dias",
            "password": ".tie5Roanl",
            "features": CMU_SAMPLE_FEATURES,
        },
        format="json",
    )
    assert genuine_res.status_code == 200
    genuine_body = genuine_res.json()
    assert genuine_body["decision"] == "GENUINE"
    assert genuine_body["confidence"] >= 0.90
    assert genuine_body["threshold"] == 0.7
    assert genuine_body["inferenceTimeMs"] > 0
    assert genuine_body["meanHoldTimeMs"] > 0
    assert genuine_body["modelVersion"] == 1

    impostor_features = {k: round(v * 2.0, 4) for k, v in CMU_SAMPLE_FEATURES.items()}
    impostor_res = client.post(
        "/api/authenticate",
        {
            "username": "thiago_dias",
            "passwordAttempt": ".tie5Roanl",
            "sample": {
                "repetition": 1,
                "features": impostor_features,
                "rawTimings": [],
            },
        },
        format="json",
    )
    assert impostor_res.status_code == 200
    impostor_body = impostor_res.json()
    assert impostor_body["decision"] == "IMPOSTOR"
    assert impostor_body["confidence"] < 0.30

    sorted_keys = sorted(CMU_SAMPLE_FEATURES.keys())
    uncertain_features = {
        k: (CMU_SAMPLE_FEATURES[k] if idx < 22 else round(CMU_SAMPLE_FEATURES[k] * 2.0, 4))
        for idx, k in enumerate(sorted_keys)
    }
    uncertain_res = client.post(
        "/api/authenticate",
        {
            "username": "thiago_dias",
            "password": ".tie5Roanl",
            "features": uncertain_features,
        },
        format="json",
    )
    assert uncertain_res.status_code == 200
    labels = list(
        KeystrokeSample.objects.filter(source=SampleSource.LOGIN)
        .order_by("id")
        .values_list("label", flat=True)
    )
    assert labels == [
        SampleLabel.GENUINE,
        SampleLabel.IMPOSTOR,
        SampleLabel.UNCERTAIN,
    ]


@pytest.mark.django_db
def test_validation_and_not_found_error_codes() -> None:
    client = APIClient()

    payload_unknown = make_section_5_3_payload(username="nao_existe", count=5)
    res_404 = client.post("/api/enrollment", payload_unknown, format="json")
    assert res_404.status_code == 404

    client.post(
        "/api/register",
        {
            "username": "user_val",
            "targetPassword": "password123",
            "confirmPassword": "password123",
        },
        format="json",
    )

    bad_enroll = make_section_5_3_payload(username="user_val", count=3)
    res_400_count = client.post("/api/enrollment", bad_enroll, format="json")
    assert res_400_count.status_code == 400

    valid_enroll = make_section_5_3_payload(username="user_val", count=5)
    assert client.post("/api/enrollment", valid_enroll, format="json").status_code == 201

    bad_features = dict(CMU_SAMPLE_FEATURES)
    bad_features.pop("H.period")
    res_400_schema = client.post(
        "/api/authenticate",
        {
            "username": "user_val",
            "password": "password123",
            "features": bad_features,
        },
        format="json",
    )
    assert res_400_schema.status_code == 400
