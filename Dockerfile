FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Cache locations inside the image (set early)
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf/transformers
ENV HUGGINGFACE_HUB_CACHE=/models/hf/hub
ENV TORCH_HOME=/models/torch
RUN mkdir -p /models/hf /models/torch

# System deps (NO ffmpeg-dev headers; we don't want to compile PyAV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip setuptools wheel

# Enforce torch stack and prevent upgrades via constraints
COPY constraints.txt /app/constraints.txt

# Re-install torch stack explicitly from cu118 index (keeps environment stable)
RUN pip install --no-cache-dir -c /app/constraints.txt \
    --index-url https://download.pytorch.org/whl/cu118 \
    torch torchaudio torchvision

# IMPORTANT:
# - Force PyAV (av) to be binary-only to avoid compiling against system ffmpeg headers.
# - Install av first so dependency resolver doesn't try to build it later.
RUN pip install --no-cache-dir -c /app/constraints.txt \
    --only-binary=:all: "av==14.0.0"

# Install app deps but keep torch pinned via constraints
RUN pip install --no-cache-dir -c /app/constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    runpod==1.7.13 requests==2.32.3 \
    whisperx pyannote.audio

# ---- Precache models (kept as its own file for clean Docker parsing) ----
COPY precache.py /app/precache.py
RUN python /app/precache.py

# Sanity check
RUN python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"

# Copy handler last (so changing handler doesn't invalidate model cache layer)
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]



