#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p native
# Pinned to the exact commits of the last verified-good build (perf validated
# on SM8845P). Update deliberately, never silently via HEAD.
LLAMA_PIN=e85e15cf6d810cd1268498c2e5b657bb3ece47bc
WHISPER_PIN=d09f61a708f3487afa956ff578e60eae5e7a233c
[ -d native/llama.cpp/.git ] || git clone https://github.com/ggml-org/llama.cpp native/llama.cpp
git -C native/llama.cpp fetch --depth 1 origin "$LLAMA_PIN" 2>/dev/null || true
git -C native/llama.cpp checkout --detach "$LLAMA_PIN" >/dev/null 2>&1
[ -d native/whisper.cpp/.git ] || git clone https://github.com/ggml-org/whisper.cpp native/whisper.cpp
git -C native/whisper.cpp fetch --depth 1 origin "$WHISPER_PIN" 2>/dev/null || true
git -C native/whisper.cpp checkout --detach "$WHISPER_PIN" >/dev/null 2>&1
printf 'llama.cpp: '; git -C native/llama.cpp rev-parse HEAD
printf 'whisper.cpp: '; git -C native/whisper.cpp rev-parse HEAD
