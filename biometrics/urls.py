from django.urls import path

from biometrics.views import AuthenticateView, EnrollmentView, HealthView, ModelsRollbackView

urlpatterns = [
    path("health", HealthView.as_view(), name="api-health"),
    path("enrollment", EnrollmentView.as_view(), name="api-enrollment"),
    path("authenticate", AuthenticateView.as_view(), name="api-authenticate"),
    path("models/rollback", ModelsRollbackView.as_view(), name="api-models-rollback"),
]
