import numpy as np
from django.db import connection
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from biometrics.ml.features import FeatureValidationError, mean_hold_time_ms
from biometrics.ml.registry import ModelIntegrityError, load_model
from biometrics.serializers import (
    AuthenticateRequestSerializer,
    EnrollmentRequestSerializer,
)
from biometrics.services import (
    AlreadyEnrolledError,
    EnrollmentValidationError,
    NotEnrolledError,
    UserNotFoundError,
    authenticate_user,
    enroll_user,
)


class HealthView(APIView):
    authentication_classes: list = []
    permission_classes: list = []

    def get(self, request: Request) -> Response:
        db_status = "ok"
        http_status = status.HTTP_200_OK
        try:
            connection.ensure_connection()
        except Exception:
            db_status = "error"
            http_status = status.HTTP_503_SERVICE_UNAVAILABLE

        return Response(
            {"status": "ok" if db_status == "ok" else "degraded", "database": db_status},
            status=http_status,
        )


class EnrollmentView(APIView):
    authentication_classes: list = []
    permission_classes: list = []

    def post(self, request: Request) -> Response:
        serializer = EnrollmentRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {
                    "success": False,
                    "message": "Payload de enrollment inválido.",
                    "errors": serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        username = serializer.validated_data.get("username")
        user_id = serializer.validated_data.get("userId")
        samples = serializer.validated_data.get("samples")
        samples_count = serializer.validated_data.get("samplesCount")

        try:
            trained_model = enroll_user(
                username=username,
                user_id=user_id,
                samples=samples,
                samples_count=samples_count,
            )
        except UserNotFoundError as exc:
            return Response(
                {"success": False, "message": str(exc)},
                status=status.HTTP_404_NOT_FOUND,
            )
        except AlreadyEnrolledError as exc:
            return Response(
                {"success": False, "message": str(exc)},
                status=status.HTTP_409_CONFLICT,
            )
        except EnrollmentValidationError as exc:
            return Response(
                {"success": False, "message": str(exc)},
                status=status.HTTP_400_BAD_REQUEST,
            )

        estimator = load_model(trained_model)
        feature_count = len(getattr(estimator, "feature_names_", []))
        samples_registered = int(trained_model.training_stats.get("genuine", 0))
        gallery = getattr(estimator, "_gallery", [])
        avg_hold_ms = (
            round(float(np.mean([mean_hold_time_ms(v.to_numpy()) for v in gallery])), 1)
            if gallery
            else 0.0
        )

        return Response(
            {
                "userId": str(trained_model.usuario.id),
                "username": trained_model.usuario.nome,
                "modelId": trained_model.id,
                "modelVersion": trained_model.versao,
                "featureCount": feature_count,
                "samplesRegistered": samples_registered,
                "calculatedAverageHoldMs": avg_hold_ms,
                "success": True,
                "message": "Enrollment concluído e modelo biométrico calibrado com sucesso.",
            },
            status=status.HTTP_201_CREATED,
        )


class AuthenticateView(APIView):
    authentication_classes: list = []
    permission_classes: list = []
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "authenticate"

    def post(self, request: Request) -> Response:
        serializer = AuthenticateRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {
                    "message": "Payload de autenticação inválido.",
                    "errors": serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        username = serializer.validated_data.get("username")
        user_id = serializer.validated_data.get("userId")
        vector = serializer.validated_data.get("resolved_vector")
        features = serializer.validated_data.get("resolved_features")

        try:
            outcome = authenticate_user(
                username=username,
                user_id=user_id,
                vector=vector,
                features=features,
            )
        except UserNotFoundError as exc:
            return Response(
                {"message": str(exc)},
                status=status.HTTP_404_NOT_FOUND,
            )
        except NotEnrolledError as exc:
            return Response(
                {"message": str(exc)},
                status=status.HTTP_409_CONFLICT,
            )
        except FeatureValidationError as exc:
            return Response(
                {"message": str(exc)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except ModelIntegrityError as exc:
            return Response(
                {"message": str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {
                "decision": outcome.decision,
                "confidence": outcome.confidence,
                "confidenceScore": outcome.confidence,
                "threshold": outcome.threshold,
                "thresholdScore": outcome.threshold,
                "inferenceTimeMs": outcome.inference_time_ms,
                "meanHoldTimeMs": outcome.mean_hold_time_ms,
                "userAverageHoldTimeMs": outcome.user_average_hold_time_ms,
                "attemptAverageHoldTimeMs": outcome.mean_hold_time_ms,
                "modelId": outcome.model_id,
                "modelVersion": outcome.model_version,
                "shouldRecordForRetraining": outcome.should_record_for_retraining,
                "message": outcome.message,
            },
            status=status.HTTP_200_OK,
        )
