# 1. USAR IMAGEN BASE MODERNA
# Usamos la imagen oficial con Python 3.11 y CUDA 12.4 para compatibilidad total
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 2. INSTALAR DEPENDENCIAS DE SISTEMA
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. PREPARAR EL ENTORNO
WORKDIR /app

# Actualizamos pip
RUN python -m pip install --upgrade pip

# 4. LIMPIEZA CRÍTICA (Evita conflictos con torch)
RUN pip uninstall -y torch torchvision torchaudio

# --- PARCHE DE SEGURIDAD PARA TU ERROR ---
# Instalamos runpod explícitamente primero para asegurar que el handler arranque
RUN pip install --no-cache-dir runpod requests

# 5. INSTALAR EL RESTO DE REQUISITOS
COPY requirements.txt /app/requirements.txt
# Intentamos instalar el resto. Si requirements.txt está vacío, esto no hará nada,
# pero al menos 'runpod' ya estará instalado por el paso anterior.
RUN pip install --no-cache-dir -r /app/requirements.txt

# 6. INSTALAR EL HANDLER
COPY handler.py /app/handler.py

# 7. COMANDO DE ARRANQUE
CMD [ "python", "-u", "/app/handler.py" ]
