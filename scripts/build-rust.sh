#!/usr/bin/env bash
# Cross-compile the Rust inference engine (rust/pocketinfer) for Android arm64
# and stage libpocketinfer.so where Gradle picks up jniLibs.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
crate="$root/rust/pocketinfer"
out_dir="${OUT_DIR:-$root/app/build/rustjni/arm64-v8a}"

if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo not found: install Rust (rustup) to build the inference engine" >&2
    exit 1
fi

ndk="${ANDROID_NDK_HOME:-}"
if [ -z "$ndk" ]; then
    for base in "${ANDROID_HOME:-$HOME/Android/Sdk}" "$HOME/Android/Sdk" /opt/android-sdk; do
        if [ -d "$base/ndk" ]; then
            ndk="$base/ndk/$(ls "$base/ndk" | sort -V | tail -1)"
            break
        fi
    done
fi
if [ -z "$ndk" ] || [ ! -d "$ndk" ]; then
    echo "Android NDK not found: set ANDROID_NDK_HOME" >&2
    exit 1
fi

host_tag="linux-x86_64"
prebuilt="$ndk/toolchains/llvm/prebuilt/$host_tag"
sysroot="$prebuilt/sysroot"
api="${ANDROID_API:-21}"
libdir="$sysroot/usr/lib/aarch64-linux-android/$api"

clang_libdir="$(find "$prebuilt" -type d -path '*clang/*/lib/linux/aarch64' 2>/dev/null | sort -V | tail -1)"
if [ -z "$clang_libdir" ]; then
    echo "libunwind.a directory not found under $prebuilt" >&2
    exit 1
fi

sysroot_opt="--sysroot=$sysroot"
lld="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin/rust-lld"
if [ ! -x "$lld" ]; then
    lld="rust-lld"
fi

export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$lld"
export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=$sysroot_opt -C link-arg=-L$libdir -C link-arg=-L$clang_libdir -C link-arg=--dynamic-linker=/system/bin/linker64 -C link-arg=$libdir/crtbegin_so.o -C link-arg=$libdir/crtend_so.o"

rustup target add aarch64-linux-android >/dev/null 2>&1 || true

cd "$crate"
cargo build --release --lib --target aarch64-linux-android
mkdir -p "$out_dir"
cp "$crate/target/aarch64-linux-android/release/libpocketinfer.so" "$out_dir/"
echo "staged: $out_dir/libpocketinfer.so"
