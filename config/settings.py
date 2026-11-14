import os
import sys
from pathlib import Path

import environ

BASE_DIR = Path(__file__).resolve().parent.parent

env = environ.Env(
    DJANGO_DEBUG=(bool, True),
    DJANGO_TEST=(bool, False),
)

env_file = BASE_DIR / ".env"
if env_file.exists():
    environ.Env.read_env(str(env_file))

SECRET_KEY = env(
    "DJANGO_SECRET_KEY",
    default="django-insecure-keystroke-biometrics-dev-secret-key",
)

DEBUG = env.bool("DJANGO_DEBUG", default=True)

ALLOWED_HOSTS: list[str] = env.list(
    "DJANGO_ALLOWED_HOSTS",
    default=["localhost", "127.0.0.1", "testserver", "api"],
)

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "biometrics",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

IS_TESTING = env.bool("DJANGO_TEST", default=False) or "pytest" in sys.modules

if IS_TESTING or not os.environ.get("DATABASE_URL"):
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }
else:
    DATABASES = {
        "default": env.db("DATABASE_URL"),
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
]

LANGUAGE_CODE = "pt-br"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
    ],
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.AllowAny",
    ],
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.ScopedRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "authenticate": env("BIOMETRICS_AUTH_THROTTLE_RATE", default="120/minute"),
    },
    "UNAUTHENTICATED_USER": None,
}

BIOMETRICS = {
    "ENROLLMENT_SAMPLES": env.int("BIOMETRICS_ENROLLMENT_SAMPLES", default=5),
    "DECISION_THRESHOLD": env.float("BIOMETRICS_DECISION_THRESHOLD", default=0.7),
    "LABEL_CONFIDENCE_MARGIN": env.float("BIOMETRICS_LABEL_CONFIDENCE_MARGIN", default=0.15),
    "RETRAIN_INTERVAL_MINUTES": env.int("BIOMETRICS_RETRAIN_INTERVAL_MINUTES", default=60),
    "RETRAIN_MIN_NEW_SAMPLES": env.int("BIOMETRICS_RETRAIN_MIN_NEW_SAMPLES", default=5),
    "RETRAIN_GENUINE_WINDOW": env.int("BIOMETRICS_RETRAIN_GENUINE_WINDOW", default=50),
    "RETRAIN_IMPOSTOR_WINDOW": env.int("BIOMETRICS_RETRAIN_IMPOSTOR_WINDOW", default=50),
    "SYNTHETIC_NEGATIVES_MIN": env.int("BIOMETRICS_SYNTHETIC_NEGATIVES_MIN", default=50),
    "SYNTHETIC_NEGATIVES_RATIO": env.int("BIOMETRICS_SYNTHETIC_NEGATIVES_RATIO", default=5),
    "M2005": {
        "threshold": env.float("BIOMETRICS_DECISION_THRESHOLD", default=0.7),
        "window_strategy": "sliding",
        "window_size": env.int("BIOMETRICS_RETRAIN_GENUINE_WINDOW", default=50),
        "threshold_method": "min_non_outlier",
        "dynamic_threshold": True,
    },
    "RANDOM_FOREST": {
        "n_estimators": 200,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": 1,
    },
    "MODELS_DIR": Path(env("BIOMETRICS_MODELS_DIR", default=str(BASE_DIR / "models_data"))),
    "POPULATION_PRIORS_PATH": BASE_DIR / "biometrics" / "data" / "population_priors.json",
}
