#!/usr/bin/env bash

# Download ESM models to the cache directory


CACHE_DIR="${1:-./esm_models_cache}"


# we are going to download only esmfold2 for now-> todo: add all models 
REPO="biohub/ESMFold2-Fast"

mkdir -p $CACHE_DIR
export HF_HOME=$CACHE_DIR
echo "HF_HOME=$HF_HOME"


read -p "Do you want to provide a Hugging Face token? [y/N]: " REPLY
if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    read -p "Paste your Hugging Face token: " TOKEN
    export HF_TOKEN=$TOKEN
    echo "HF_TOKEN=$HF_TOKEN"
else
    echo "Skipping Hugging Face token setup."
fi

echo "Downloading $REPO..."

module load python 

mamba activate esm  

python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$REPO')
print('[OK] ESM model downloaded')
print(f'Cache populated: $CACHE_DIR')
"

echo "Download complete."

echo "Model cache ready: $CACHE_DIR"

