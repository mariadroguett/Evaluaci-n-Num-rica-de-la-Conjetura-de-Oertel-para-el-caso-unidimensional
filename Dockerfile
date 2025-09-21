FROM python:3.10-slim-bullseye

# Configuración básica de Python y límites de hilos para BLAS/OpenMP
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_MAX_THREADS=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Primero dependencias para aprovechar la caché de Docker
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Comando por defecto (se puede sobrescribir con `docker run ... <cmd>`)
CMD ["python", "-u", "main.py"]
