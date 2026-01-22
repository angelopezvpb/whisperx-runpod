FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

# IMPORTANT:
# Keep torch/torchaudio/torchvision exactly as in base image (cu118),
# and pin versions that are known to work together.
#
# 1) Install runpod + requests for downloading audio locally
# 2) Install whisperx pinned
# 3) Install pyannote.audio pinned (avoid latest surprises)
#
# Note: If whisperx pulls incompatible deps, pip will resolve; pins protect you.

RUN pip install --no-cache-dir -U pip setuptools wheel

# Core runtime deps
RUN pip install --no-cache-dir \
    runpod==1.7.13 \
    requests==2.32.3

# WhisperX + diarization stack (PINNED)
RUN pip install --no-cache-dir \
    "whisperx==3.1.5" \
    "pyannote.audio==3.1.1"

# (Optional but recommended) enforce torchaudio/torchvision to match torch from base image
# Check torch version in base image first; for runpod/pytorch:2.1.0-cu118 it's typically torch==2.1.0+cu118
RUN python - << 'EOF'\n\
import torch\n\
print('torch:', torch.__version__)\n\
EOF

# Copy handler
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
