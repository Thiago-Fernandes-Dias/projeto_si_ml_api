from typing import Any

from rest_framework import serializers


class RegisterRequestSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150, trim_whitespace=True)
    password = serializers.CharField(
        required=False,
        allow_blank=False,
        write_only=True,
    )
    targetPassword = serializers.CharField(
        required=False,
        allow_blank=False,
        write_only=True,
    )
    confirmPassword = serializers.CharField(
        required=False,
        allow_blank=False,
        write_only=True,
    )

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        password = attrs.get("password") or attrs.get("targetPassword")
        if not password:
            raise serializers.ValidationError(
                {"password": "O campo 'password' (ou 'targetPassword') é obrigatório."}
            )
        confirm = attrs.get("confirmPassword")
        if confirm is not None and confirm != password:
            raise serializers.ValidationError(
                {"confirmPassword": "A confirmação de senha não confere."}
            )
        attrs["resolved_password"] = password
        return attrs


class EnrollmentRequestSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150, trim_whitespace=True)
    passwordLength = serializers.IntegerField(required=False, min_value=1)
    samplesCount = serializers.IntegerField(required=False, min_value=1)
    samples = serializers.ListField(
        child=serializers.DictField(),
        allow_empty=False,
    )


class AuthenticateRequestSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150, trim_whitespace=True)
    password = serializers.CharField(required=False, allow_blank=True, write_only=True)
    passwordAttempt = serializers.CharField(required=False, allow_blank=True, write_only=True)
    features = serializers.DictField(required=False)
    rawTimings = serializers.JSONField(required=False, allow_null=True)
    sample = serializers.DictField(required=False)

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        password = attrs.get("password")
        if password is None:
            password = attrs.get("passwordAttempt")
        if password is None:
            raise serializers.ValidationError(
                {"password": "O campo 'password' (ou 'passwordAttempt') é obrigatório."}
            )

        sample_obj = attrs.get("sample")
        features = attrs.get("features")
        raw_timings = attrs.get("rawTimings")

        if features is None and isinstance(sample_obj, dict):
            features = sample_obj.get("features")
            if raw_timings is None:
                raw_timings = sample_obj.get("rawTimings", sample_obj.get("raw_timings"))

        if not isinstance(features, dict) or len(features) == 0:
            raise serializers.ValidationError(
                {"features": "O dicionário 'features' (ou 'sample.features') é obrigatório."}
            )

        attrs["resolved_password"] = password
        attrs["resolved_features"] = features
        attrs["resolved_raw_timings"] = raw_timings
        return attrs
