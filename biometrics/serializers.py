from typing import Any

from rest_framework import serializers


class EnrollmentRequestSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=50, required=False, trim_whitespace=True)
    userId = serializers.UUIDField(required=False)
    vectors = serializers.ListField(
        child=serializers.ListField(child=serializers.FloatField()),
        allow_empty=False,
    )

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        if not attrs.get("username") and not attrs.get("userId"):
            raise serializers.ValidationError(
                {"username": "Informe 'username' ou 'userId'."}
            )
        return attrs


class AuthenticateRequestSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=50, required=False, trim_whitespace=True)
    userId = serializers.UUIDField(required=False)
    vector = serializers.ListField(
        child=serializers.FloatField(),
        allow_empty=False,
    )

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        if not attrs.get("username") and not attrs.get("userId"):
            raise serializers.ValidationError(
                {"username": "Informe 'username' ou 'userId'."}
            )
        return attrs
