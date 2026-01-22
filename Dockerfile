# syntax=docker/dockerfile:1.4
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# ---- caches persistentes (para que el build no sea eterno) ----
# HuggingFace / Torch caches dentro de la imagen
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf/transformers
ENV HUGGINGFACE_HUB_CACHE=/models/hf/hub
ENV TORCH_HOME=/models/torch
RUN mkdir -p /models/hf /models/torch

# ---- deps del sistema (runtime) ----
# OJO: NO metemos libav*-dev ni cosas para compilar PyAV: queremos wheel binario.
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# pip + build basics
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip setuptools wheel

# ---- constraints (evitar que te suba torch a 2.6/2.8) ----
COPY constraints.txt /app/constraints.txt

# ---- 1) instalar PyAV como BINARIO sí o sí (si no hay wheel, fallará aquí y lo ves rápido) ----
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --only-binary=:all: "av==11.0.0"

# ---- 2) instalar whisperx/pyannote/runpod respetando constraints ----
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -c /app/constraints.txt \
      runpod==1.7.13 requests==2.32.3 \
      whisperx==3.1.5 pyannote.audio==3.1.1

# ---- sanity check rápido ----
RUN python -c "import torch, torchaudio; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('devices', torch.cuda.device_count())"

# ---- precache (se mantiene en capa separada) ----
COPY precache.py /app/precache.py
RUN python /app/precache.py

# Handler al final (cambiar handler NO invalida capas pesadas)
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]




