# 1. USAR IMAGEN BASE MODERNA (CUDA 12.1+ es necesario para Torch 2.8)
# Usamos una imagen de RunPod estable con Python 3.10 y CUDA 12
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# 2. INSTALAR DEPENDENCIAS DE SISTEMA (FFmpeg y Git son obligatorios para WhisperX)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. PREPARAR EL ENTORNO
WORKDIR /app

# Actualizamos pip para asegurar compatibilidad con las ruedas modernas
RUN python -m pip install --upgrade pip

# 4. LIMPIEZA CRÍTICA: Desinstalar el Torch base
# Esto evita el conflicto "The conflict is caused by... torch". 
# Borramos lo que trae la imagen para instalar TU lista limpia.
RUN pip uninstall -y torch torchvision torchaudio

# 5. COPIAR E INSTALAR TU LISTA MAESTRA
COPY requirements.txt /app/requirements.txt

# Instalamos usando tu lista. El flag --no-cache-dir reduce el peso.
RUN pip install --no-cache-dir -r /app/requirements.txt

# 6. INSTALAR EL HANDLER DE RUNPOD
# Asumiendo que tienes un script python para arrancar
ADD src . 
# Ajusta el nombre de tu archivo si no es handler.py
CMD [ "python", "-u", "/handler.py" ]



