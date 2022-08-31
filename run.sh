#! /usr/bin/env bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
source "${SCRIPT_DIR}/.env"
NAME='simple-transformers'
mkdir -p "$HUGGINGFACE_CACHE_DIR"
mkdir -p "$TORCH_EXTENSIONS_HOST_DIR"
mkdir -p "$CHECKPOINT_DIR"
docker build -t "$NAME" .
docker run --rm -it \
  --gpus "\"device=$1\"" \
  --shm-size=5g \
  --env-file .env \
  -h="$(hostname -s)" \
  -e TERM=xterm-256color \
  -p "$JUPYTER_PORT:$JUPYTER_PORT" \
  --mount "src=$SCRIPT_DIR,target=/src,type=bind" \
  --mount "src=$CHECKPOINT_DIR,target=/checkpoints,type=bind" \
  --mount "src=$HUGGINGFACE_CACHE_DIR,target=/root/.cache/huggingface/,type=bind" \
  --mount "src=$TORCH_EXTENSIONS_HOST_DIR,target=/root/.cache/torch_extensions/,type=bind" \
  "$NAME" "${@:2}"
