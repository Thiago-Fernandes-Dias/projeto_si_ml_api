from django.db import connection
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from biometrics.ml.features import FeatureValidationError
from biometrics.ml.registry import ModelIntegrityError
from biometrics.serializers import (
    AuthenticateRequestSerializer,
    EnrollmentRequestSerializer,
    RegisterRequestSerializer,
)
from biometrics.services import (
    AlreadyEnrolledError,
    EnrollmentValidationError,
    InvalidCredentialsError,
    NotEnrolledError,
    UserAlreadyExistsError,
    UserNotFoundError,
    authenticate_user,
    enroll_user,
    register_user,
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


class RegisterView(APIView):
    authentication_classes: list = []
    permission_classes: list = []

    def post(self, request: Request) -> Response:
        serializer = RegisterRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {
                    "success": False,
                    "message": "Dados de cadastro inválidos.",
                    "errors": serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        username = serializer.validated_data["username"]
        password = serializer.validated_data["resolved_password"]

        try:
            user = register_user(username=username, password=password)
        except UserAlreadyExistsError as exc:
            return Response(
                {"success": False, "message": str(exc)},
                status=status.HTTP_409_CONFLICT,
            )
        except ValueError as exc:
            return Response(
                {"success": False, "message": str(exc)},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response(
            {
                "username": user.username,
                "success": True,
                "message": "Usuário cadastrado com sucesso.",
            },
            status=status.HTTP_201_CREATED,
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

        username = serializer.validated_data["username"]
        samples = serializer.validated_data["samples"]
        samples_count = serializer.validated_data.get("samplesCount")

        try:
            trained_model = enroll_user(
                username=username,
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

        feature_count = len(trained_model.user.biometric_profile.feature_names)
        return Response(
            {
                "username": trained_model.user.username,
                "modelVersion": trained_model.version,
                "featureCount": feature_count,
                "samplesRegistered": len(samples),
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

        username = serializer.validated_data["username"]
        password = serializer.validated_data["resolved_password"]
        features = serializer.validated_data["resolved_features"]
        raw_timings = serializer.validated_data["resolved_raw_timings"]

        try:
            outcome = authenticate_user(
                username=username,
                password=password,
                features=features,
                raw_timings=raw_timings,
            )
        except InvalidCredentialsError:
            return Response(
                {
                    "decision": "INVALID_CREDENTIALS",
                    "message": "Credenciais inválidas.",
                },
                status=status.HTTP_401_UNAUTHORIZED,
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
                "modelVersion": outcome.model_version,
                "message": outcome.message,
            },
            status=status.HTTP_200_OK,
        )
