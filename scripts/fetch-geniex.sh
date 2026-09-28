#!/usr/bin/env bash
# Populate npubench jniLibs/assets from the private HuggingFace dataset.
#
# The GenieX llama.cpp runtime is BSD-3, but the prebuilt binaries are not
# committed to the repository: they are fetched here so the public repo carries
# no licensed artifacts. The QNN libraries are deliberately not included, since
# this test only exercises the ggml-hexagon (FastRPC) path.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
mod="$root/npubench"
libs="$mod/src/main/jniLibs/arm64-v8a"
assets="$mod/src/main/assets"
tarball="${TARBALL:-/tmp/opencode/gx/geniex-llamacpp-arm64.tar.gz}"
hf_repo="${HF_REPO:-Aleinto/qairt-sdk}"
hf_file="${HF_FILE:-geniex-llamacpp-arm64.tar.gz}"

mkdir -p "$libs" "$assets"
rm -rf "$libs"/*.so "$assets"/geniex-bench

if [ ! -s "$tarball" ]; then
  echo "fetching $hf_file from $hf_repo"
  token="${HF_TOKEN:?set HF_TOKEN or place the tarball at $tarball}"
  curl -sL --fail -H "Authorization: Bearer $token" -o "$tarball" \
    "https://huggingface.co/datasets/$hf_repo/resolve/main/$hf_file"
fi

staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT
tar xzf "$tarball" -C "$staging"
cp "$staging"/lib/*.so "$libs"/
cp "$staging"/bin/geniex-bench "$assets"/geniex-bench
chmod +x "$assets"/geniex-bench

echo "installed $(ls "$libs" | wc -l) libraries and the bench asset"
ls -la "$libs" | tail -4
