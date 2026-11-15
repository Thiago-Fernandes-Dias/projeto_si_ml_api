import pytest
from rest_framework.test import APIClient

from biometrics.models import KeystrokeSample, TrainedModel
from tests.conftest import (
    CMU_SAMPLE_FEATURES,
    create_user_in_db,
    make_genuine_feature_dicts,
    make_section_5_3_payload,
)


@pytest.mark.django_db
def test_health_endpoint() -> None:
    client = APIClient()
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


@pytest.mark.django_db
def test_register_endpoint_is_not_exposed() -> None:
    client = APIClient()
    response = client.post(
        "/api/register",
        {"username": "thiago_dias"},
        format="json",
    )
    assert response.status_code == 404


@pytest.mark.django_db
def test_full_flow_enrollment_and_authenticate() -> None:
    client = APIClient()
    user = create_user_in_db(username="thiago_dias")

    pre_auth = client.post(
        "/api/authenticate",
        {
            "username": "thiago_dias",
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

    user.refresh_from_db()
    assert TrainedModel.objects.filter(usuario=user).count() == 1
    assert KeystrokeSample.objects.filter(usuario=user).count() == 0
    assert user.biometria_cadastrada_em is None

    dup_enroll = client.post("/api/enrollment", enroll_payload, format="json")
    assert dup_enroll.status_code == 409

    unknown_user = client.post(
        "/api/authenticate",
        {
            "username": "ghost_user",
            "features": CMU_SAMPLE_FEATURES,
        },
        format="json",
    )
    assert unknown_user.status_code == 404

    genuine_res = client.post(
        "/api/authenticate",
        {
            "username": "thiago_dias",
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
    assert genuine_body["userAverageHoldTimeMs"] > 0
    assert genuine_body["modelVersion"] == 1
    assert genuine_body["shouldRecordForRetraining"] is True

    borderline_features = {
        k: (round(v * 2.0, 4) if idx < 6 else v)
        for idx, (k, v) in enumerate(CMU_SAMPLE_FEATURES.items())
    }
    borderline_res = client.post(
        "/api/authenticate",
        {
            "username": "thiago_dias",
            "features": borderline_features,
        },
        format="json",
    )
    assert borderline_res.status_code == 200
    borderline_body = borderline_res.json()
    assert borderline_body["decision"] == "GENUINE"
    threshold_val = borderline_body["threshold"]
    assert threshold_val <= borderline_body["confidence"] < threshold_val + 0.15
    assert borderline_body["shouldRecordForRetraining"] is False

    impostor_features = {k: round(v * 2.0, 4) for k, v in CMU_SAMPLE_FEATURES.items()}
    impostor_res = client.post(
        "/api/authenticate",
        {
            "username": "thiago_dias",
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
    assert impostor_body["shouldRecordForRetraining"] is False

    out_of_order_features = {k: CMU_SAMPLE_FEATURES[k] for k in sorted(CMU_SAMPLE_FEATURES)}
    out_of_order_res = client.post(
        "/api/authenticate",
        {
            "username": "thiago_dias",
            "features": out_of_order_features,
        },
        format="json",
    )
    assert out_of_order_res.status_code == 400
    assert KeystrokeSample.objects.filter(usuario=user).count() == 0


@pytest.mark.django_db
def test_enrollment_and_authenticate_with_backend_positional_vectors() -> None:
    client = APIClient()
    user = create_user_in_db(username="backend_user")

    vectors = [list(d.values()) for d in make_genuine_feature_dicts(count=5, seed=7)]
    enroll_res = client.post(
        "/api/enrollment",
        {
            "username": "backend_user",
            "userId": str(user.id),
            "vectors": vectors,
        },
        format="json",
    )
    assert enroll_res.status_code == 201
    data = enroll_res.json()
    assert data["modelVersion"] == 1
    assert data["featureCount"] == len(vectors[0])
    assert data["samplesRegistered"] == 5
    assert KeystrokeSample.objects.filter(usuario=user).count() == 0

    auth_res = client.post(
        "/api/authenticate",
        {
            "username": "backend_user",
            "userId": str(user.id),
            "vector": vectors[0],
        },
        format="json",
    )
    assert auth_res.status_code == 200
    auth_body = auth_res.json()
    assert auth_body["decision"] == "GENUINE"
    assert auth_body["shouldRecordForRetraining"] is True

    impostor_vec = [round(v * 2.0, 4) for v in vectors[0]]
    impostor_res = client.post(
        "/api/authenticate",
        {
            "username": "backend_user",
            "userId": str(user.id),
            "vector": impostor_vec,
        },
        format="json",
    )
    assert impostor_res.status_code == 200
    impostor_vec_body = impostor_res.json()
    assert impostor_vec_body["decision"] == "IMPOSTOR"
    assert impostor_vec_body["shouldRecordForRetraining"] is False

    wrong_dim_res = client.post(
        "/api/authenticate",
        {
            "username": "backend_user",
            "userId": str(user.id),
            "vector": vectors[0][:-3],
        },
        format="json",
    )
    assert wrong_dim_res.status_code == 400

    non_cmu_len_res = client.post(
        "/api/authenticate",
        {
            "username": "backend_user",
            "userId": str(user.id),
            "vector": vectors[0][:-1],
        },
        format="json",
    )
    assert non_cmu_len_res.status_code == 400

    invalid_slot_vec = list(vectors[0])
    invalid_slot_vec[0] = -0.05
    invalid_slot_res = client.post(
        "/api/authenticate",
        {
            "username": "backend_user",
            "userId": str(user.id),
            "vector": invalid_slot_vec,
        },
        format="json",
    )
    assert invalid_slot_res.status_code == 400
    assert KeystrokeSample.objects.filter(usuario=user).count() == 0


@pytest.mark.django_db
def test_validation_and_not_found_error_codes() -> None:
    client = APIClient()

    payload_unknown = make_section_5_3_payload(username="nao_existe", count=5)
    res_404 = client.post("/api/enrollment", payload_unknown, format="json")
    assert res_404.status_code == 404

    create_user_in_db(username="user_val")

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
            "features": bad_features,
        },
        format="json",
    )
    assert res_400_schema.status_code == 400
