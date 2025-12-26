# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Entrainement du modele pendant le build pour garantir sa presence
RUN python model.py

# Lancement avec Gunicorn (Serveur WSGI Production)
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8501", "app_flask:app"]
