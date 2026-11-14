from django.contrib.auth.models import User
from django.db import models


class BiometricProfile(models.Model):
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name="biometric_profile",
    )
    feature_names = models.JSONField()
    enrolled_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"BiometricProfile({self.user.username}, features={len(self.feature_names)})"


class SampleSource(models.TextChoices):
    ENROLLMENT = "ENROLLMENT", "Enrollment"
    LOGIN = "LOGIN", "Login"


class SampleLabel(models.TextChoices):
    GENUINE = "GENUINE", "Genuine"
    IMPOSTOR = "IMPOSTOR", "Impostor"
    UNCERTAIN = "UNCERTAIN", "Uncertain"


class TrainingTrigger(models.TextChoices):
    ENROLLMENT = "ENROLLMENT", "Enrollment"
    SCHEDULED = "SCHEDULED", "Scheduled"
    MANUAL = "MANUAL", "Manual"


class TrainedModel(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="trained_models",
    )
    version = models.PositiveIntegerField()
    file_path = models.CharField(max_length=500)
    sha256 = models.CharField(max_length=64)
    sklearn_version = models.CharField(max_length=20)
    trigger = models.CharField(choices=TrainingTrigger.choices, max_length=16)
    training_stats = models.JSONField()
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["user", "version"],
                name="unique_model_version",
            ),
            models.UniqueConstraint(
                fields=["user"],
                condition=models.Q(is_active=True),
                name="single_active_model",
            ),
        ]

    def __str__(self) -> str:
        status = "active" if self.is_active else "inactive"
        return f"TrainedModel({self.user.username}, v{self.version}, {status})"


class KeystrokeSample(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="keystroke_samples",
    )
    features = models.JSONField()
    raw_timings = models.JSONField(null=True, blank=True)
    source = models.CharField(choices=SampleSource.choices, max_length=16)
    label = models.CharField(choices=SampleLabel.choices, max_length=16)
    genuine_probability = models.FloatField(null=True, blank=True)
    classified_by = models.ForeignKey(
        TrainedModel,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
    )
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    def __str__(self) -> str:
        return f"KeystrokeSample({self.user.username}, {self.source}, {self.label})"
