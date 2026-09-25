FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем локальные wheels и устанавливаем их вместе с matplotlib
COPY wheels/ /app/wheels/
RUN pip install --no-cache-dir --find-links=/app/wheels \
    onnxruntime \
    opencv-python-headless \
    numpy \
    scipy \
    pyyaml \
    matplotlib \
    mediapipe==0.10.14

# Копируем код и веса
COPY volumetric_skull_generator.py /app/volumetric_skull_generator.py
COPY weights/ /app/weights/
COPY render_utils/ /app/render_utils/

# Компилируем C-рендерер прямо на месте
RUN mkdir -p /app/render_utils/asset && cd /app/render_utils && gcc -shared -Wall -O3 render.c -o asset/render.so -fPIC

ENTRYPOINT ["python", "/app/volumetric_skull_generator.py"]