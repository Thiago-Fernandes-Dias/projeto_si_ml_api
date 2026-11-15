import logging

import numpy as np
from django.db import connection
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from biometrics.logging_utils import log_structured
from biometrics.ml.features import FeatureValidationError, mean_hold_time_ms
from biometrics.ml.registry import ModelIntegrityError, load_model
from biometrics.serializers import (
    AuthenticateRequestSerializer,
    EnrollmentRequestSerializer,
    RollbackModelsRequestSerializer,
)
from biometrics.services import (
    AlreadyEnrolledError,
    EnrollmentValidationError,
    NotEnrolledError,
    UserNotFoundError,
    authenticate_user,
    enroll_user,
    rollback_models_after,
)

logger = logging.getLogger(__name__)


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

        log_structured(
            logger,
            logging.DEBUG,
            "ml_api.health.checked",
            "Verificação de saúde da ML API executada.",
            flow="health",
            database=db_status,
            httpStatus=http_status,
        )
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
            log_structured(
                logger,
                logging.DEBUG,
                "ml_api.enrollment.payload_invalid",
                "Payload de enrollment rejeitado pela validação.",
                flow="enrollment",
                errorFields=sorted(serializer.errors.keys()),
            )
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
        vectors = serializer.validated_data["vectors"]

        try:
            trained_model = enroll_user(
                username=username,
                user_id=user_id,
                vectors=vectors,
            )
        except UserNotFoundError as exc:
            log_structured(
                logger,
                logging.DEBUG,
                "ml_api.enrollment.user_not_found",
                "Usuário não encontrado durante enrollment.",
                flow="enrollment",
                username=username,
                userId=str(user_id) if user_id else None,
            )
            return Response(
                {"success": False, "message": str(exc)},
                status=status.HTTP_404_NOT_FOUND,
            )
        except AlreadyEnrolledError as exc:
            log_structured(
                logger,
                logging.DEBUG,
                "ml_api.enrollment.already_enrolled",
                "Enrollment duplicado rejeitado.",
                flow="enrollment",
                username=username,
                userId=str(user_id) if user_id else None,
            )
            return Response(
                {"success": False, "message": str(exc)},
                status=status.HTTP_409_CONFLICT,
            )
        except EnrollmentValidationError as exc:
            log_structured(
                logger,
                logging.DEBUG,
                "ml_api.enrollment.validation_failed",
                "Validação biométrica falhou durante enrollment.",
                flow="enrollment",
                username=username,
                reason=str(exc),
            )
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

        log_structured(
            logger,
            logging.DEBUG,
            "ml_api.enrollment.succeeded",
            "Enrollment concluído com sucesso na ML API.",
            flow="enrollment",
            userId=str(trained_model.usuario.id),
            username=trained_model.usuario.nome,
            modelId=trained_model.id,
            modelVersion=trained_model.versao,
            featureCount=feature_count,
            samplesRegistered=samples_registered,
            calculatedAverageHoldMs=avg_hold_ms,
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
            log_structured(
                logger,
                logging.DEBUG,
                "ml_api.login.payload_invalid",
                "Payload de autenticação rejeitado pela validação.",
                flow="login",
                errorFields=sorted(serializer.errors.keys()),
            )
            return Response(
                {
                    "message": "Payload de autenticação inválido.",
                    "errors": serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        username = serializer.validated_data.get("username")
        user_id = serializer.validated_data.get("userId")
        vector = serializer.validated_data["vector"]

        try:
            outcome = authenticate_user(
                username=username,
                user_id=user_id,
                vector=vector,
            )
        except UserNotFoundError as exc:
            log_structured(
                logger,
                logging.DEBUG,
                "ml_api.login.user_not_found",
                "Usuário não encontrado na autenticação.",
                flow="login",
                username=username,
                userId=str(user_id) if user_id else None,
            )
            return Response(
                {"message": str(exc)},
                status=status.HTTP_404_NOT_FOUND,
            )
        except NotEnrolledError as exc:
            log_structured(
                logger,
                logging.DEBUG,
                "ml_api.login.not_enrolled",
                "Usuário ainda sem modelo biométrico ativo.",
                flow="login",
                username=username,
                userId=str(user_id) if user_id else None,
            )
            return Response(
                {"message": str(exc)},
                status=status.HTTP_409_CONFLICT,
            )
        except FeatureValidationError as exc:
            log_structured(
                logger,
                logging.DEBUG,
                "ml_api.login.feature_validation_failed",
                "Vetor biométrico inválido na autenticação.",
                flow="login",
                username=username,
                reason=str(exc),
            )
            return Response(
                {"message": str(exc)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except ModelIntegrityError as exc:
            log_structured(
                logger,
                logging.ERROR,
                "ml_api.login.model_integrity_error",
                "Falha de integridade do modelo ao autenticar.",
                flow="login",
                username=username,
                error=str(exc),
            )
            return Response(
                {"message": str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        log_structured(
            logger,
            logging.DEBUG,
            "ml_api.login.responded",
            "Inferência biométrica concluída na ML API.",
            flow="login",
            username=username,
            userId=str(user_id) if user_id else None,
            decision=outcome.decision,
            confidence=round(outcome.confidence, 4),
            threshold=round(outcome.threshold, 4),
            inferenceTimeMs=round(outcome.inference_time_ms, 2),
            modelId=outcome.model_id,
            modelVersion=outcome.model_version,
            shouldRecordForRetraining=outcome.should_record_for_retraining,
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


class ModelsRollbackView(APIView):
    authentication_classes: list = []
    permission_classes: list = []

    def post(self, request: Request) -> Response:
        serializer = RollbackModelsRequestSerializer(data=request.data)
        if not serializer.is_valid():
            log_structured(
                logger,
                logging.DEBUG,
                "ml_api.models.rollback_payload_invalid",
                "Payload de rollback de modelos rejeitado.",
                flow="password_change",
                errorFields=sorted(serializer.errors.keys()),
            )
            return Response(
                {
                    "success": False,
                    "message": "Payload de rollback inválido.",
                    "errors": serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            outcome = rollback_models_after(
                created_after=serializer.validated_data["createdAfter"],
                username=serializer.validated_data.get("username"),
                user_id=serializer.validated_data.get("userId"),
            )
        except UserNotFoundError as exc:
            return Response(
                {"success": False, "message": str(exc)},
                status=status.HTTP_404_NOT_FOUND,
            )

        return Response(
            {
                "success": True,
                "deletedCount": outcome.deleted_count,
                "activeVersion": outcome.active_version,
            },
            status=status.HTTP_200_OK,
        )
