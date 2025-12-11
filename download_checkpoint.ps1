# Download Pop2Piano model checkpoint from HuggingFace

$CheckpointDir = "checkpoints"
$CheckpointFile = "model.ckpt"
$CheckpointUrl = "https://huggingface.co/sweetcocoa/pop2piano/resolve/main/model-1999-val_0.67311615.ckpt?download=true"

if (-not (Test-Path $CheckpointDir)) {
    New-Item -ItemType Directory -Path $CheckpointDir | Out-Null
}

Write-Host "Downloading Pop2Piano checkpoint..."
Invoke-WebRequest -Uri $CheckpointUrl -OutFile "$CheckpointDir\$CheckpointFile"

Write-Host "Done! Checkpoint saved to $CheckpointDir\$CheckpointFile"
