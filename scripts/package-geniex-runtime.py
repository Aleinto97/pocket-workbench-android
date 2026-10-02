#!/usr/bin/env python3
"""Stage the GenieX/llama.cpp runtime for the main app APK.

The app already ships whisper.cpp's ggml libraries, and GenieX ships its own
copies with the same sonames. Android packages one file per name, so the
GenieX copies are renamed and every DT_NEEDED entry is rewritten to match.
The DSP skel keeps its exact name: RPCCode looks it up by absolute URI.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "npubench/src/main/jniLibs/arm64-v8a"
DEST = ROOT / "app/build/geniexjni/arm64-v8a"

# whisper.cpp ships these under the same sonames in the app APK.
RENAMED = {
    "libggml-base.so": "libgeniex-ggml-base.so",
    "libggml-cpu.so": "libgeniex-ggml-cpu.so",
    "libggml.so": "libgeniex-ggml.so",
}
# Native libraries the app already owns; nothing may depend on these names.
RESERVED = {"libggml-base.so", "libggml-cpu.so", "libggml.so", "libparakeet.so",
            "libpocketnative.so", "libwhisper.so"}
# Bionic/platform libraries are provided by the device, never staged by us.
PLATFORM = {"libc.so", "libm.so", "libdl.so", "liblog.so", "libz.so", "libstdc++.so",
            "libgcc.so", "libc++_shared.so", "libc++_static.so", "libandroid.so", "libEGL.so",
            "libGLESv3.so", "libOpenSLES.so", "libaaudio.so", "libvulkan.so", "libOpenCL.so"}


def run(*args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def needed(path):
    out = subprocess.run(["readelf", "-d", str(path)], check=True, capture_output=True, text=True).stdout
    return [line.split("[")[1].rstrip("]") for line in out.splitlines() if "(NEEDED)" in line]


def main():
    if not SOURCE.is_dir():
        raise SystemExit("GenieX runtime missing. Run scripts/fetch-geniex.sh first.")
    shutil.rmtree(DEST.parent, ignore_errors=True)
    DEST.mkdir(parents=True)
    for source in sorted(SOURCE.glob("*.so")):
        shutil.copyfile(source, DEST / RENAMED.get(source.name, source.name))
    for old, new in RENAMED.items():
        run("patchelf", "--set-soname", new, DEST / new)
        run("patchelf", "--remove-rpath", DEST / new)
    for library in sorted(DEST.glob("*.so")):
        for required in needed(library):
            target = RENAMED.get(required, required)
            if required in RESERVED and required not in RENAMED:
                raise SystemExit(f"{library.name} still depends on app-owned {required}")
            run("patchelf", "--replace-needed", required, target, library)
    staged = {p.name for p in DEST.glob("*.so")}
    for library in sorted(DEST.glob("*.so")):
        for required in needed(library):
            if required not in staged and required not in PLATFORM:
                raise SystemExit(f"{library.name} needs unstaged {required}")
    manifest = {"libraries": sorted(staged), "renamed": RENAMED}
    (DEST.parent / "geniexjni.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"GenieX runtime staged: {len(staged)} libraries, {sum(p.stat().st_size for p in DEST.glob('*.so')) // 1048576} MiB")


if __name__ == "__main__":
    main()