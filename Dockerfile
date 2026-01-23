FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# Cache de HF dentro de la imagen (para que el precache se quede en layers)
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf/transformers
ENV HUGGINGFACE_HUB_CACHE=/models/hf/hub
ENV TORCH_HOME=/models/torch

RUN mkdir -p /models/hf /models/torch

# --- System deps (solo lo necesario para compilar av==10.0.0 contra ffmpeg 4.2) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg pkg-config build-essential python3-dev \
    libavcodec-dev libavformat-dev libavdevice-dev libavutil-dev libavfilter-dev \
    libswscale-dev libswresample-dev \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip setuptools wheel

# --- Constraints para impedir que whisperx/pyannote te actualicen torch a 2.6/2.8 ---
COPY constraints.txt /app/constraints.txt

# Asegura que torch stack cu118 se quede como toca (si ya está, esto es rápido)
RUN pip install --no-cache-dir -c /app/constraints.txt \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  torch torchaudio torchvision

# 1) Instala PyAV compatible con FFmpeg 4.2 (evita el AV_CODEC_CAP_OTHER_THREADS)
RUN pip install --no-cache-dir "av==10.0.0"

# 2) Instala deps de app (con constraints para que NO toque torch)
RUN pip install --no-cache-dir -c /app/constraints.txt \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  runpod==1.7.13 requests==2.32.5 \
  whisperx==3.7.2 pyannote.audio==3.4.0

# --- Precache (sin heredoc) ---
COPY precache.py /app/precache.py
RUN python /app/precache.py

# Sanity check
RUN python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"

# Copia handler al final para no invalidar layers pesados en cambios de código
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]



