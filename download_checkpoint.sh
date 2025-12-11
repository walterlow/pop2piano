#!/bin/bash
# Download Pop2Piano model checkpoint from HuggingFace

CHECKPOINT_DIR="checkpoints"
CHECKPOINT_FILE="model.ckpt"
CHECKPOINT_URL="https://huggingface.co/sweetcocoa/pop2piano/resolve/main/model-1999-val_0.67311615.ckpt?download=true"

mkdir -p "$CHECKPOINT_DIR"

echo "Downloading Pop2Piano checkpoint..."
curl -L -o "$CHECKPOINT_DIR/$CHECKPOINT_FILE" "$CHECKPOINT_URL"

echo "Done! Checkpoint saved to $CHECKPOINT_DIR/$CHECKPOINT_FILE"
