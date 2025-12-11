# syntax=docker/dockerfile:1
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fluidsynth \
    libsndfile1 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir yt-dlp && \
    ln -s $(which yt-dlp) /usr/local/bin/youtube-dl

# Copy application code (for production builds)
COPY . .

# Default command (can be overridden)
CMD ["python", "--version"]
