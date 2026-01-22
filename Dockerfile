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

# Pre-download Whisper large-v3 + try to cache alignment models for many languages
RUN python - << 'EOF'
import whisperx

whisperx.load_model("large-v3", "cpu", compute_type="int8", language=None)
print("✅ Cached Whisper large-v3")

langs = [
  "en","es","fr","de","it","pt","nl","pl","cs","sk","sl","hr","sr","bs","hu","ro","bg","ru","uk","el","tr",
  "ar","he","fa","ur","hi","bn","ta","te","kn","ml","mr","gu","pa",
  "zh","ja","ko","th","vi","id","ms","tl",
  "sv","no","da","fi","is",
  "et","lv","lt"
]

ok, fail = 0, 0
for l in langs:
  try:
    whisperx.load_align_model(language_code=l, device="cpu")
    ok += 1
    print(f"✅ Cached align: {l}")
  except Exception as e:
    fail += 1
    print(f"⚠️ Skip align {l}: {e}")

print(f"Done. align ok={ok} fail={fail}")
EOF

# Sanity check
RUN python - << 'EOF'
import torch, torchaudio, torchvision
print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
EOF

COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]

