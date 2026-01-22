FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# System deps for ffmpeg + building PyAV (av) when needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git pkg-config build-essential python3-dev \
    libavcodec-dev libavformat-dev libavdevice-dev libavutil-dev libavfilter-dev \
    libswscale-dev libswresample-dev \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip setuptools wheel

# 1) Enforce torch stack (CU118) and prevent upgrades via constraints
COPY constraints.txt /app/constraints.txt

RUN pip install --no-cache-dir -c /app/constraints.txt \
    --index-url https://download.pytorch.org/whl/cu118 \
    torch torchaudio torchvision

# 2) Install app deps but keep torch pinned (constraints applies here too)
RUN pip install --no-cache-dir -c /app/constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    runpod==1.7.13 requests==2.32.3 \
    whisperx pyannote.audio

# Quick sanity check (fails build if something tries to upgrade torch)
RUN python - << 'EOF'
import torch, torchaudio, torchvision
print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
EOF

COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
