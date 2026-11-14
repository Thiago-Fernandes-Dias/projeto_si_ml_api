from django.urls import path

from biometrics.views import AuthenticateView, EnrollmentView, HealthView, RegisterView

urlpatterns = [
    path("health", HealthView.as_view(), name="api-health"),
    path("register", RegisterView.as_view(), name="api-register"),
    path("enrollment", EnrollmentView.as_view(), name="api-enrollment"),
    path("authenticate", AuthenticateView.as_view(), name="api-authenticate"),
]
