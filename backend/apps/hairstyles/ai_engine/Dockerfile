FROM python:3.10-slim

# Install system dependencies compatible with modern Debian base images
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install stable dependencies inside the container
# Устанавливаем PyTorch с дополнительным индексом PyPI для недостающих зависимостей
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple/

# Добавляем onnxruntime вместе с остальными библиотеками
RUN pip install --no-cache-dir opencv-python-headless mediapipe==0.10.14 numpy pyyaml onnxruntime

WORKDIR /app

COPY volumetric_skull_generator.py /app/volumetric_skull_generator.py

ENTRYPOINT ["python", "/app/volumetric_skull_generator.py"]