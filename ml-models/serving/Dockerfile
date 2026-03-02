FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY config.py .
COPY models/ models/
COPY serving/ serving/
COPY data/ data/
COPY pipeline/ pipeline/
COPY evaluation/ evaluation/
COPY agent_integration.py .

# Copy trained models (mount as volume in production)
# COPY trained_models/ trained_models/

EXPOSE 8001

ENV ML_SERVER_PORT=8001

CMD ["python", "-m", "serving.model_server"]
