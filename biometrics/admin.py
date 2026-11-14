from django.contrib import admin

from biometrics.models import BiometricProfile, KeystrokeSample, TrainedModel


@admin.register(BiometricProfile)
class BiometricProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "feature_count", "enrolled_at")
    search_fields = ("user__username",)
    readonly_fields = ("enrolled_at",)

    @admin.display(description="Features")
    def feature_count(self, obj: BiometricProfile) -> int:
        return len(obj.feature_names or [])


@admin.register(KeystrokeSample)
class KeystrokeSampleAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "user",
        "source",
        "label",
        "genuine_probability",
        "classified_by",
        "created_at",
    )
    list_filter = ("source", "label", "created_at")
    search_fields = ("user__username",)
    readonly_fields = ("created_at",)


@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ("user", "version", "trigger", "is_active", "sklearn_version", "created_at")
    list_filter = ("trigger", "is_active", "created_at")
    search_fields = ("user__username", "sha256")
    readonly_fields = ("sha256", "sklearn_version", "created_at")
