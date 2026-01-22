FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# Cache locations inside the image (set early)
ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf/transformers
ENV HUGGINGFACE_HUB_CACHE=/models/hf/hub
ENV TORCH_HOME=/models/torch
RUN mkdir -p /models/hf /models/torch

# System deps for ffmpeg + building PyAV (av) when needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git pkg-config build-essential python3-dev \
    libffi-dev libssl-dev \
    libavcodec-dev libavformat-dev libavdevice-dev libavutil-dev libavfilter-dev \
    libswscale-dev libswresample-dev \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip setuptools wheel

# Enforce torch stack (CU118) and prevent upgrades via constraints
COPY constraints.txt /app/constraints.txt

RUN pip install --no-cache-dir -c /app/constraints.txt \
    --index-url https://download.pytorch.org/whl/cu118 \
    torch torchaudio torchvision

# Install app deps but keep torch pinned
RUN pip install --no-cache-dir -c /app/constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    runpod==1.7.13 requests==2.32.3 \
    whisperx pyannote.audio

# ---- Precache models (NO heredoc; use a script) ----
COPY precache.py /app/precache.py
RUN python /app/precache.py

# Sanity check
RUN python -c "import torch, torchaudio, torchvision; \
print('torch:', torch.__version__); \
print('torchaudio:', torchaudio.__version__); \
print('torchvision:', torchvision.__version__); \
print('cuda available:', torch.cuda.is_available())"

# Copy handler last (so changing handler doesn't invalidate model cache layer)
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]


