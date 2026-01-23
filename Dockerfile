# syntax=docker/dockerfile:1.5
FROM runpod/pytorch:2.4.0-py3.11-cuda12.1.0-devel

WORKDIR /app

# --- Caches HF dentro de la imagen (para precache opcional) ---
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf/transformers
ENV HUGGINGFACE_HUB_CACHE=/models/hf/hub
ENV TORCH_HOME=/models/torch

RUN mkdir -p /models/hf /models/torch

# --- System deps mínimos (NO build toolchain) ---
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir -U pip setuptools wheel

# Importante: copia constraints antes para que cachee bien
COPY constraints.txt /app/constraints.txt

# 1) Fuerza PyAV por WHEEL (cero compilación)
#    (si no existe wheel compatible, fallará aquí rápido y no después de 10 minutos)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --only-binary=:all: -c /app/constraints.txt \
      "av==15.1.0"

# 2) Instala stack app (pinned) sin que torch se actualice
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -c /app/constraints.txt \
      "runpod==1.7.13" \
      "requests==2.32.5" \
      "whisperx==3.7.2" \
      "pyannote.audio==3.4.0"

# (Opcional) Precache de modelos: descomentarlo cuando ya no estés iterando cada 2 min
# COPY precache.py /app/precache.py
# RUN --mount=type=cache,target=/root/.cache/pip \
#     python /app/precache.py

# Sanity check rápido
RUN python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

# Copia handler al final para NO invalidar capas pesadas al tocar código
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]



