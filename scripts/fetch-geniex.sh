#!/usr/bin/env bash
# Populate npubench jniLibs from the private HuggingFace dataset.
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
# keep the download inside a directory that exists on both the workstation and CI
tarball="${TARBALL:-${TMPDIR:-/tmp}/geniex-llamacpp-arm64.tar.gz}"
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
# The benchmark is an ELF program, but Android only extracts lib*.so from
# jniLibs and marks the extracted file executable. App data dirs on this
# device are mounted noexec, so that is the only place it can be run from.
# Naming it like a library is deliberate.
cp "$staging"/bin/geniex-bench "$libs"/libgeniexbench.so

# A one-shot benchmark process can exit after writing its report: the prebuilt
# runtime sometimes segfaults while unloading the Android plugin at deinit.
# Compile our tiny exit shim for the same ABI (not a fetched binary).
if [ -n "${BENCH_CC:-}" ]; then
  cc="$BENCH_CC"
elif [ -n "${ANDROID_NDK_HOME:-}" ]; then
  cc="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang"
elif command -v aarch64-linux-android-clang >/dev/null 2>&1; then
  cc="$(command -v aarch64-linux-android-clang)"
else
  echo "FATAL: set ANDROID_NDK_HOME or BENCH_CC to an Android arm64 C compiler" >&2
  exit 1
fi
"$cc" --target=aarch64-linux-android29 -O2 -shared -fPIC \
  -Wl,-soname,libgeniexbench_exit.so \
  -o "$libs/libgeniexbench_exit.so" "$mod/src/main/c/bench_exit.c"

echo "installed $(ls "$libs" | wc -l) libraries including the benchmark"
ls -la "$libs" | tail -4
