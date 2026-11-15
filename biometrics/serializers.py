from typing import Any

from rest_framework import serializers


class EnrollmentRequestSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=50, required=False, trim_whitespace=True)
    userId = serializers.UUIDField(required=False)
    passwordLength = serializers.IntegerField(required=False, min_value=1)
    samplesCount = serializers.IntegerField(required=False, min_value=1)
    samples = serializers.ListField(
        child=serializers.JSONField(),
        required=False,
        allow_empty=True,
    )
    vectors = serializers.ListField(
        child=serializers.ListField(child=serializers.FloatField()),
        required=False,
        allow_empty=True,
    )

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        if not attrs.get("username") and not attrs.get("userId"):
            raise serializers.ValidationError(
                {"username": "Informe 'username' ou 'userId'."}
            )
        if attrs.get("samples") is None and attrs.get("vectors") is not None:
            attrs["samples"] = attrs["vectors"]
        return attrs


class AuthenticateRequestSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=50, required=False, trim_whitespace=True)
    userId = serializers.UUIDField(required=False)
    vector = serializers.ListField(
        child=serializers.FloatField(),
        required=False,
        allow_empty=False,
    )
    features = serializers.DictField(required=False)
    sample = serializers.DictField(required=False)

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        if not attrs.get("username") and not attrs.get("userId"):
            raise serializers.ValidationError(
                {"username": "Informe 'username' ou 'userId'."}
            )

        sample_obj = attrs.get("sample")
        vector = attrs.get("vector")
        features = attrs.get("features")

        if isinstance(sample_obj, dict):
            if vector is None and isinstance(sample_obj.get("vector"), list):
                vector = sample_obj.get("vector")
            if features is None and isinstance(sample_obj.get("features"), dict):
                features = sample_obj.get("features")

        has_vector = isinstance(vector, list) and len(vector) > 0
        has_features = isinstance(features, dict) and len(features) > 0

        if not (has_vector or has_features):
            raise serializers.ValidationError(
                {"features": "Informe 'vector' ou 'features'."}
            )

        attrs["resolved_vector"] = vector if has_vector else None
        attrs["resolved_features"] = features if has_features else None
        return attrs
