# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    fluidsynth \
    libsndfile1 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .

# Install PyTorch with CUDA 11.8
RUN pip install --no-cache-dir torch==2.0.1+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir yt-dlp && \
    ln -s $(which yt-dlp) /usr/local/bin/youtube-dl

# Copy application code (for production builds)
COPY . .

# Expose the web server port
EXPOSE 8000

# Default command: run the FastAPI server
CMD ["python", "app.py"]
