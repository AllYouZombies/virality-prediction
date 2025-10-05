FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

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
    curl \
    && python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# СНАЧАЛА устанавливаем PyTorch nightly
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# ЗАТЕМ устанавливаем остальные зависимости
# PyTorch уже установлен, поэтому зависимости не будут тянуть стабильную версию
RUN --mount=type=bind,source=requirements.txt,target=/app/requirements.txt \
    --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY app/ /app/app/
COPY ViViT/ /app/ViViT/

RUN mkdir -p /app/models /app/uploads

COPY create_demo_model.py /app/

RUN echo '#!/bin/bash\n\
if [ ! -f /app/models/vivit_model.pth ]; then\n\
    echo "Модель не найдена, создаем демо-модель..."\n\
    python3 /app/create_demo_model.py\n\
fi\n\
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000' > /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
