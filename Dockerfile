FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /app

# System deps needed to BUILD PyAV (av) + runtime ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    pkg-config \
    build-essential \
    python3-dev \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir -U pip setuptools wheel

# Instala whisperx + diarization stack
RUN pip install --no-cache-dir \
    whisperx \
    pyannote.audio \
    runpod

COPY handler.py /app/

CMD ["python", "-u", "handler.py"]

