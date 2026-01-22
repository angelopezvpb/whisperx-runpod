FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# Cache locations inside the image (set early)
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf/transformers
ENV HUGGINGFACE_HUB_CACHE=/models/hf/hub
ENV TORCH_HOME=/models/torch
ENV HF_HUB_DISABLE_TELEMETRY=1
RUN mkdir -p /models/hf /models/torch

# ✅ Solo runtime deps (NO dev headers, NO build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip setuptools wheel

COPY constraints.txt /app/constraints.txt

# ✅ Fuerza torch CU118 EXACTO (evita upgrades raros)
RUN pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu118 \
  -c /app/constraints.txt \
  torch torchvision torchaudio

# ✅ CRÍTICO: fuerza PyAV wheel (si no hay wheel, falla aquí y no compila)
RUN pip install --no-cache-dir \
  --only-binary=:all: \
  -c /app/constraints.txt \
  av==11.0.0

# ✅ App deps (con constraints para que NO toque torch/av)
RUN pip install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  -c /app/constraints.txt \
  runpod requests whisperx pyannote.audio

# ---- Precache models ----
COPY precache.py /app/precache.py
RUN python /app/precache.py

# Sanity check
RUN python -c "import torch, torchaudio, torchvision; \
print('torch:', torch.__version__); \
print('torchaudio:', torchaudio.__version__); \
print('torchvision:', torchvision.__version__); \
print('cuda available:', torch.cuda.is_available()); \
print('device count:', torch.cuda.device_count())"

# Copy handler last (para que cambiar handler NO invalide capas pesadas)
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]


