# ==========================================
# ETAPA 1: BUILDER
# ==========================================
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Instalamos herramientas básicas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Creamos el entorno virtual
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Instalamos dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# --- LIMPIEZA SEGURA (SOLO BASURA REAL) ---
# Solo borramos cache (__pycache__) y tests.
# ¡YA NO TOCAMOS .dist-info NI .so!
RUN find /opt/venv -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyc" -delete && \
    find /opt/venv -type d -name "tests" -exec rm -r {} + 2>/dev/null || true

# ==========================================
# ETAPA 2: RUNTIME
# ==========================================
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8080

WORKDIR /app

# Librerías para OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiamos el entorno sano desde el builder
COPY --from=builder /opt/venv /opt/venv

# Creamos usuario seguro
RUN useradd -m -u 1000 myuser

# Copiamos código y modelo
COPY --chown=myuser:myuser ./app ./app
COPY --chown=myuser:myuser ./models ./models

USER myuser

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]