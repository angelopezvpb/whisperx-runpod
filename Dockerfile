# 1. USAR IMAGEN BASE MODERNA
FROM runpod/pytorch:2.2.1-py3.11-cuda12.1.1-devel-ubuntu22.04

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

# 4. LIMPIEZA CRÍTICA
RUN pip uninstall -y torch torchvision torchaudio

# 5. INSTALAR REQUISITOS (Asegúrate de haber subido requirements.txt)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 6. INSTALAR EL HANDLER (PARCHEADO)
# Cambiamos 'ADD src .' por 'COPY' directo para evitar el error "src not found"
# si tu archivo está en la raíz del repo.
COPY handler.py /app/handler.py

# 7. COMANDO DE ARRANQUE (PARCHEADO)
# Ajustamos la ruta para que apunte a /app/handler.py en lugar de la raíz /
CMD [ "python", "-u", "/app/handler.py" ]

