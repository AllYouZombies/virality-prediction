FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl

# Устанавливаем pipenv
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pipenv

# Pipenv правильно разрешает зависимости с учетом источников
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=./Pipfile.lock,target=/app/Pipfile.lock \
    PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy --ignore-pipfile

# Активируем виртуальное окружение pipenv
ENV PATH="/app/.venv/bin:$PATH"

COPY app/ /app/app/

RUN mkdir -p /app/models

EXPOSE 8000
