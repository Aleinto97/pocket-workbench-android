#!/usr/bin/env bash
# Fetch the Linux-module host-side binaries: a static PRoot and a static curl.
#
# Both are public upstream releases, verified by SHA-256 below. They are staged
# under tools/linux-bin/ (gitignored) and packaged into the APK by
# scripts/package-linux-runtime.py as lib*.so files, the same trick the GenieX
# benchmark uses: Android only extracts lib*.so from jniLibs as executables.
#
# The Debian rootfs itself is NOT fetched here: at ~35 MB it is downloaded
# on-device, on demand, by the app (LinuxModule), with the same SHA-256 check.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/tools/linux-bin"
mkdir -p "$dest"

# ahmed-alnassif/proot v26.08.25-7266fb3, NDK-built static proot for Android.
# Ran on the reference tablet (proot --version OK, Debian trixie boots, apt works).
proot_url="https://github.com/ahmed-alnassif/proot/releases/download/v26.08.25-7266fb3/proot-aarch64.zip"
proot_sha256="045ced0a516d0ec419e11ff8cafdb641fb490ea49fc747704fedecc88cab1155"
# stunnel/static-curl 8.22.0, fully static aarch64 curl with OpenSSL.
# Ran on the reference tablet (curl --version + HTTPS OK).
curl_url="https://github.com/stunnel/static-curl/releases/download/8.22.0/curl-linux-aarch64-glibc-8.22.0.tar.xz"
curl_sha256="fa4de50f80fb2fbbf77a7bf8385891b6cb1a501a2e28b1ed18cf25e11fd66b66"
curl_inner_sha256="45b2fa110164e548ced7ec5535ad7f26f70d086e2af3d973e500b0986edf5ec3"

have() { command -v "$1" >/dev/null 2>&1; }
need() { have "$1" || { echo "FATAL: missing required tool: $1" >&2; exit 1; }; }
need curl; need sha256sum; need unzip; need tar

fetch() { # url sha256 dest
  if [ -s "$3" ] && [ "$(sha256sum "$3" | cut -d' ' -f1)" = "$2" ]; then
    echo "cached: $3"
    return 0
  fi
  echo "fetching $1"
  curl -sL --fail --retry 3 -o "$3.tmp" "$1"
  actual="$(sha256sum "$3.tmp" | cut -d' ' -f1)"
  if [ "$actual" != "$2" ]; then
    echo "FATAL: SHA-256 mismatch for $1 (got $actual)" >&2
    rm -f "$3.tmp"
    exit 1
  fi
  mv -f "$3.tmp" "$3"
}

fetch "$proot_url" "$proot_sha256" "$dest/proot-aarch64.zip"
fetch "$curl_url" "$curl_sha256" "$dest/curl-linux-aarch64.tar.xz"

rm -rf "$dest/stage" && mkdir -p "$dest/stage"
unzip -o -q "$dest/proot-aarch64.zip" -d "$dest/stage/proot"
tar -xf "$dest/curl-linux-aarch64.tar.xz" -C "$dest/stage"
mv -f "$dest/stage/proot/proot" "$dest/proot-aarch64"
mv -f "$dest/stage/curl" "$dest/curl-aarch64"
rm -rf "$dest/stage"
actual_inner="$(sha256sum "$dest/curl-aarch64" | cut -d' ' -f1)"
if [ "$actual_inner" != "$curl_inner_sha256" ]; then
  echo "FATAL: inner curl SHA-256 mismatch (got $actual_inner)" >&2
  exit 1
fi
chmod +x "$dest/proot-aarch64" "$dest/curl-aarch64"
echo "linux host binaries ready in $dest:"
ls -la "$dest/proot-aarch64" "$dest/curl-aarch64"
