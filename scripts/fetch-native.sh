#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p native
[ -d native/llama.cpp/.git ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp native/llama.cpp
[ -d native/whisper.cpp/.git ] || git clone --depth 1 https://github.com/ggml-org/whisper.cpp native/whisper.cpp
printf 'llama.cpp: '; git -C native/llama.cpp rev-parse HEAD
printf 'whisper.cpp: '; git -C native/whisper.cpp rev-parse HEAD
