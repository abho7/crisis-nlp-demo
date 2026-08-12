# Works unchanged on Hugging Face Spaces (Docker SDK) and Render.
# HF Spaces expects 7860; Render injects $PORT. Default covers the former.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
# The trained stand-in is ~0.5 MB, so it ships in the image rather than being
# fetched at boot. Retraining is `python training/train_standin.py`.
COPY artifacts/standin_model.joblib ./artifacts/

EXPOSE 7860

# Shell form so $PORT expands at runtime (Render injects it; HF Spaces uses the
# 7860 default above). `exec` replaces the shell with uvicorn so SIGTERM reaches
# the server directly -- without it the platform's stop signal hits /bin/sh and
# the container waits out the kill timeout on every deploy.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
