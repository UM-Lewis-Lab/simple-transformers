#! /usr/bin/env bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
name='simple_transformers'
docker build -t "$name" .
docker run --rm -it \
	--gpus "\"device=$1\"" \
	--shm-size=5g \
	--env-file .env \
	-h="$(hostname -s)" \
	-e TERM=xterm-256color \
	--mount "src=$SCRIPT_DIR,target=/src,type=bind" \
	--mount "src=$HUGGINGFACE_CACHE_DIR,target=/root/.cache/huggingface/,type=bind" \
	--mount "src=$TORCH_EXTENSIONS_HOST_DIR,target=/root/.cache/torch_extensions/,type=bind" \
	"$name" "${@:2}"
