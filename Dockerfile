FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG GID=1000
ARG UID=1000

WORKDIR /opt

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

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=./requirements.txt,target=/opt/requirements.txt \
    pip install --break-system-packages --no-cache-dir -r requirements.txt

RUN (groupadd -g ${GID} group || true) && \
    (useradd -m -u ${UID} -g group user || true) && \
    chown -R ${UID}:${GID} /opt

COPY --chown=appuser:appgroup app/ /opt/app/

RUN mkdir -p /opt/models && chown -R ${UID}:${GID} /opt/models

USER ${UID}

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
