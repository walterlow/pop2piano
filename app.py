"""
Pop2Piano FastAPI Server

Provides a web interface for generating piano covers from audio files or YouTube URLs.
Supports job queue with background processing.
"""

import os
import uuid
import subprocess
import shutil
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import Optional
from enum import Enum

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from omegaconf import OmegaConf

from transformer_wrapper import TransformerWrapper

# Configuration
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "./data/uploads"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./data/outputs"))
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "./checkpoints/model.ckpt")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "./config.yaml")

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="Pop2Piano", description="Generate piano covers from any song")

# Global model (loaded on startup)
model = None

# Job queue and storage
job_queue = queue.Queue()
jobs = {}  # job_id -> JobInfo
jobs_lock = threading.Lock()


class JobStatus(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobInfo:
    def __init__(self, job_id: str, job_type: str, composer: str):
        self.job_id = job_id
        self.job_type = job_type  # "youtube" or "upload"
        self.composer = composer
        self.status = JobStatus.QUEUED
        self.progress = "Waiting in queue..."
        self.error = None
        self.created_at = datetime.now()
        self.original_audio = None
        self.piano_audio = None
        self.midi = None


class YouTubeRequest(BaseModel):
    url: str
    composer: str = "composer1"


class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: str


class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: str
    error: Optional[str] = None
    original_audio: Optional[str] = None
    piano_audio: Optional[str] = None
    midi: Optional[str] = None


class QueueStatusResponse(BaseModel):
    queue_length: int
    processing: Optional[str] = None


def load_model():
    """Load the Pop2Piano model."""
    global model

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Model will not be loaded. Please provide a valid checkpoint.")
        return None

    config = OmegaConf.load(CONFIG_PATH)
    model = TransformerWrapper.load_from_checkpoint(CHECKPOINT_PATH, config=config)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    print(f"Model loaded from {CHECKPOINT_PATH}")
    return model


def update_job(job_id: str, **kwargs):
    """Thread-safe job update."""
    with jobs_lock:
        if job_id in jobs:
            for key, value in kwargs.items():
                setattr(jobs[job_id], key, value)


def download_youtube_audio(url: str, output_path: str, job_id: str) -> str:
    """Download audio from YouTube URL using yt-dlp."""
    update_job(job_id, status=JobStatus.DOWNLOADING, progress="Downloading from YouTube...")

    try:
        cmd = [
            "yt-dlp",
            "-x",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", output_path,
            "--postprocessor-args", "ffmpeg:-ac 1 -ar 22050",
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            raise Exception(f"yt-dlp failed: {result.stderr}")

        output_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]

        for ext in [".wav", ".mp3", ".m4a", ".webm"]:
            potential_path = os.path.join(output_dir, base_name + ext)
            if os.path.exists(potential_path):
                if ext != ".wav":
                    wav_path = os.path.join(output_dir, base_name + ".wav")
                    subprocess.run([
                        "ffmpeg", "-y", "-i", potential_path,
                        "-ac", "1", "-ar", "22050", wav_path
                    ], capture_output=True)
                    os.remove(potential_path)
                    return wav_path
                return potential_path

        raise Exception("Downloaded file not found")

    except subprocess.TimeoutExpired:
        raise Exception("Download timed out")


def process_audio(audio_path: str, composer: str, job_id: str):
    """Process audio file and generate piano cover."""
    update_job(job_id, status=JobStatus.PROCESSING, progress="Analyzing audio and generating piano cover...")

    if model is None:
        raise Exception("Model not loaded")

    output_job_dir = OUTPUT_DIR / job_id
    output_job_dir.mkdir(parents=True, exist_ok=True)

    # Copy original audio to output
    original_path = output_job_dir / "original.wav"
    shutil.copy(audio_path, original_path)

    # Generate piano cover
    midi_path = output_job_dir / "piano.mid"
    piano_wav_path = output_job_dir / "piano.wav"

    update_job(job_id, progress="Running AI model...")

    # Generate MIDI only (no stereo mix)
    pm, used_composer, _, _ = model.generate(
        audio_path=str(audio_path),
        composer=composer,
        save_midi=True,
        save_mix=False,
        midi_path=str(midi_path),
    )

    update_job(job_id, progress="Rendering piano audio...")

    # Render MIDI to piano-only audio using fluidsynth
    sample_rate = 22050
    piano_audio = pm.fluidsynth(fs=sample_rate)

    import soundfile as sf
    sf.write(str(piano_wav_path), piano_audio, sample_rate)

    return {
        "original_audio": f"/api/file/{job_id}/original.wav",
        "piano_audio": f"/api/file/{job_id}/piano.wav",
        "midi": f"/api/file/{job_id}/piano.mid"
    }


def process_job(job_id: str, job_data: dict):
    """Process a single job (runs in worker thread)."""
    try:
        job_type = job_data["type"]
        composer = job_data["composer"]

        if job_type == "youtube":
            url = job_data["url"]
            upload_job_dir = UPLOAD_DIR / job_id
            upload_job_dir.mkdir(parents=True, exist_ok=True)
            audio_path = upload_job_dir / "input.wav"

            downloaded_path = download_youtube_audio(url, str(audio_path), job_id)
            result = process_audio(downloaded_path, composer, job_id)

        elif job_type == "upload":
            audio_path = job_data["audio_path"]
            result = process_audio(audio_path, composer, job_id)

        update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress="Complete!",
            original_audio=result["original_audio"],
            piano_audio=result["piano_audio"],
            midi=result["midi"]
        )

    except Exception as e:
        update_job(
            job_id,
            status=JobStatus.FAILED,
            progress="Failed",
            error=str(e)
        )


def worker():
    """Background worker that processes jobs from the queue."""
    while True:
        job_id, job_data = job_queue.get()
        try:
            process_job(job_id, job_data)
        except Exception as e:
            update_job(job_id, status=JobStatus.FAILED, error=str(e))
        finally:
            job_queue.task_done()


@app.on_event("startup")
async def startup_event():
    """Load model and start worker on startup."""
    load_model()

    # Start background worker thread
    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    print("Background worker started")


@app.get("/")
async def root():
    """Serve the frontend."""
    return FileResponse("frontend/index.html")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available(),
        "queue_length": job_queue.qsize()
    }


@app.get("/api/queue", response_model=QueueStatusResponse)
async def get_queue_status():
    """Get current queue status."""
    processing = None
    with jobs_lock:
        for job_id, job in jobs.items():
            if job.status in [JobStatus.DOWNLOADING, JobStatus.PROCESSING]:
                processing = job_id
                break

    return {
        "queue_length": job_queue.qsize(),
        "processing": processing
    }


@app.post("/api/process/youtube", response_model=JobResponse)
async def process_youtube(request: YouTubeRequest):
    """Queue a YouTube URL for processing."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    job_id = str(uuid.uuid4())[:8]

    # Create job
    with jobs_lock:
        jobs[job_id] = JobInfo(job_id, "youtube", request.composer)

    # Queue job
    job_data = {
        "type": "youtube",
        "url": request.url,
        "composer": request.composer
    }
    job_queue.put((job_id, job_data))

    position = job_queue.qsize()

    return {
        "job_id": job_id,
        "status": JobStatus.QUEUED,
        "progress": f"Queued (position {position})"
    }


@app.post("/api/process/upload", response_model=JobResponse)
async def process_upload(
    file: UploadFile = File(...),
    composer: str = Form("composer1")
):
    """Queue an uploaded audio file for processing."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    job_id = str(uuid.uuid4())[:8]

    # Validate file type
    allowed_extensions = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Create upload directory and save file
    upload_job_dir = UPLOAD_DIR / job_id
    upload_job_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_job_dir / f"input{file_ext}"

    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Convert to WAV if needed
    if file_ext != ".wav":
        wav_path = upload_job_dir / "input.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(input_path),
            "-ac", "1", "-ar", "22050", str(wav_path)
        ], capture_output=True, check=True)
        audio_path = str(wav_path)
    else:
        audio_path = str(input_path)

    # Create job
    with jobs_lock:
        jobs[job_id] = JobInfo(job_id, "upload", composer)

    # Queue job
    job_data = {
        "type": "upload",
        "audio_path": audio_path,
        "composer": composer
    }
    job_queue.put((job_id, job_data))

    position = job_queue.qsize()

    return {
        "job_id": job_id,
        "status": JobStatus.QUEUED,
        "progress": f"Queued (position {position})"
    }


@app.get("/api/status/{job_id}", response_model=StatusResponse)
async def get_job_status(job_id: str):
    """Get status of a job."""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job = jobs[job_id]
        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "error": job.error,
            "original_audio": job.original_audio,
            "piano_audio": job.piano_audio,
            "midi": job.midi
        }


@app.get("/api/file/{job_id}/{filename}")
async def get_file(job_id: str, filename: str):
    """Serve a generated file."""
    file_path = OUTPUT_DIR / job_id / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    ext = file_path.suffix.lower()
    media_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".mid": "audio/midi",
        ".midi": "audio/midi"
    }
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(file_path, media_type=media_type)


@app.get("/api/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download a generated file."""
    file_path = OUTPUT_DIR / job_id / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        filename=filename,
        media_type="application/octet-stream"
    )


@app.get("/api/composers")
async def get_composers():
    """Get available composer styles."""
    if model is None:
        return {
            "composers": [
                {"id": f"composer{i}", "name": f"Style {i}"}
                for i in range(1, 22)
            ]
        }

    return {
        "composers": [
            {"id": k, "name": k.replace("composer", "Style ")}
            for k in model.composer_to_feature_token.keys()
        ]
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("DEBUG", "false").lower() == "true"

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=debug
    )
