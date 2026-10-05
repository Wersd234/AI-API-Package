# AI-RP-Proxy — two-stage RP inference relay
# Slim single-stage image; config comes from environment variables at runtime.

FROM python:3.12-slim

# No .pyc clutter, and unbuffered stdout so progress logs show up live
# in `docker logs` / `docker compose logs -f`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps first so code edits reuse the cached layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (keep mock_backends.py out of the production image —
# it is a dev tool, shipped in the repo for local testing).
COPY main.py pipeline.py config.py schemas.py ./
COPY stage2_prompt.txt ./
COPY .env.example ./

# The .env is deliberately NOT baked in: supply it at runtime via
# `docker run --env-file .env ...` or compose's `env_file:`.

EXPOSE 5000

CMD ["python", "main.py"]