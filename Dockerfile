# 1. USAR IMAGEN BASE MODERNA (Con Python 3.11 para compatibilidad con contourpy/whisperx)
# Usamos una etiqueta oficial que incluye PyTorch 2.4 y CUDA 12.4
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 2. INSTALAR DEPENDENCIAS DE SISTEMA
# Instalamos git y ffmpeg necesarios para el procesamiento de audio
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. PREPARAR EL ENTORNO
WORKDIR /app

# Actualizamos pip a la última versión para evitar errores de compilación de wheels
RUN python -m pip install --upgrade pip

# 4. LIMPIEZA CRÍTICA
# Eliminamos el PyTorch que trae la imagen para evitar conflictos con tu requirements.txt
RUN pip uninstall -y torch torchvision torchaudio

# 5. INSTALAR REQUISITOS
# Copiamos tu lista maestra. IMPORTANTE: El archivo requirements.txt debe estar en el repo.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 6. INSTALAR EL HANDLER
# Copiamos el script de arranque directamente a la carpeta de la app
COPY handler.py /app/handler.py

# 7. COMANDO DE ARRANQUE
# Ejecutamos el handler. El flag -u permite ver los logs en tiempo real en RunPod.
CMD [ "python", "-u", "/app/handler.py" ]
