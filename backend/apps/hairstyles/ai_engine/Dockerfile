##FROM python:3.10-slim
##
##RUN apt-get update && apt-get install -y \
##    build-essential \
##    libgl1 \
##    libglib2.0-0 \
##    && rm -rf /var/lib/apt/lists/*
##
##WORKDIR /appx
##
### 1. Тяжелые wheels и зависимости (кэшируются намертво и не пересобираются)
##COPY wheels/ /app/wheels/
##RUN pip install --no-cache-dir --find-links=/app/wheels \
##    onnxruntime \
##    opencv-python-headless \
##    numpy \
##    scipy \
##    pyyaml \
##    matplotlib \
##    mediapipe==0.10.14
##
### Копируем код и веса
##COPY volumetric_skull_generator.py /app/volumetric_skull_generator.py
##COPY weights/ /app/weights/
##COPY render_utils/ /app/render_utils/
##
### Компилируем C-рендерер прямо на месте
##RUN mkdir -p /app/render_utils/asset && cd /app/render_utils && gcc -shared -Wall -O3 render.c -o asset/render.so -fPIC
##
##ENTRYPOINT ["python", "/app/volumetric_skull_generator.py"]
#
#FROM python:3.10-slim
#
#RUN apt-get update && apt-get install -y \
#    build-essential \
#    libgl1 \
#    libglib2.0-0 \
#    && rm -rf /var/lib/apt/lists/*
#
#WORKDIR /app
#
## 1. Тяжелые зависимости и неизменяемые веса (кэшируются намертво)
#COPY wheels/ /app/wheels/
#RUN pip install --no-cache-dir --find-links=/app/wheels \
#    onnxruntime \
#    opencv-python-headless \
#    numpy \
#    scipy \
#    pyyaml \
#    matplotlib \
#    mediapipe==0.10.14
#
#COPY weights/ /app/weights/
#
## 2. Папка render_utils (активно меняется, кэш сбрасывается при правках)
#COPY render_utils/ /app/render_utils/
#
## 3. Компиляция C-рендерера
#RUN mkdir -p /app/render_utils/asset && \
#    cd /app/render_utils && \
#    gcc -shared -Wall -O3 render.c -o asset/render.so -fPIC
#
## 4. Основной скрипт генератора
#COPY volumetric_skull_generator.py /app/volumetric_skull_generator.py
#
#ENTRYPOINT ["python", "/app/volumetric_skull_generator.py"]
#=================================================



FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Тяжелые зависимости и веса (надежно кэшируются первыми)
#COPY wheels/ /app/wheels/
#RUN pip install --no-cache-dir --find-links=/app/wheels \
#    onnxruntime \
#    opencv-python-headless \
#    numpy \
#    scipy \
#    pyyaml \
#    matplotlib \
#    mediapipe==0.10.14
COPY wheels/ /app/wheels/
RUN pip install --no-cache-dir --find-links=/app/wheels \
    onnxruntime==1.12.0 \
    opencv-python-headless==4.7.0.72 \
    numpy==1.23.5 \
    scipy==1.9.3 \
    pyyaml \
    matplotlib \
    mediapipe==0.10.14

COPY weights/ /app/weights/

# 2. Копируем оставшийся чистый код (благодаря .dockerignore папки detectors, generators и т.д. не попадут внутрь)
COPY . /app/

# 3. Компиляция C-рендерера
RUN mkdir -p /app/render_utils/asset && \
    cd /app/render_utils && \
    gcc -shared -Wall -O3 render.c -o asset/render.so -fPIC

ENTRYPOINT ["python", "/app/volumetric_skull_generator.py"]