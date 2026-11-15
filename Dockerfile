FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY config ./config
COPY biometrics ./biometrics
COPY manage.py ./

RUN pip install --upgrade pip && pip install .

RUN mkdir -p /app/models_data

EXPOSE 8000

CMD ["sh", "-c", "python manage.py migrate --fake-initial --noinput && gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 2 --timeout 60 --error-logfile -"]
