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

EXPOSE 8501

# Le fichier .streamlit/config.toml sera copie automatiquement avec COPY . .
CMD ["streamlit", "run", "app.py"]
