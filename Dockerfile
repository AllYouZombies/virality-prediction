# virality-prediction/Dockerfile (обновленная версия)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch nightly с CUDA 12.4 для поддержки новых GPU (compute capability 12.0)
RUN pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124 --no-deps && \
    pip install numpy typing-extensions sympy networkx jinja2 fsspec filelock pillow

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY app/ /app/app/
COPY ViViT/ /app/ViViT/

# Создаем необходимые директории
RUN mkdir -p /app/models /app/uploads

# Скрипт для создания демо-модели если нужно
COPY create_demo_model.py /app/

# Создаем entrypoint скрипт
RUN echo '#!/bin/bash\n\
if [ ! -f /app/models/vivit_model.pth ]; then\n\
    echo "Модель не найдена, создаем демо-модель..."\n\
    python3 /app/create_demo_model.py\n\
fi\n\
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000' > /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
