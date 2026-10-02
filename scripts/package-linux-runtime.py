#!/usr/bin/env python3
"""Stage the Linux-module executables for the main app APK.

proot and curl ship as lib*.so under jniLibs so the package manager extracts
them as executables (same trick as the GenieX benchmark): app data dirs on
the reference device are mounted noexec, so this is the only place they can
be run from. Both binaries only need platform libraries (libc/libdl) or are
fully static, so no DT_NEEDED rewriting is required — but it is verified.

The Debian rootfs is deliberately NOT staged here: it is downloaded on-device
on demand (see LinuxModule / filesDir/linux).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools/linux-bin"
DEST = ROOT / "app/build/linuxjni/arm64-v8a"

# Binary name on disk -> name inside the APK (lib*.so so it stays executable).
STAGED = {
    "proot-aarch64": "libproot.so",
    "curl-aarch64": "libpocketcurl.so",
}

PLATFORM = {"libc.so", "libm.so", "libdl.so", "liblog.so", "libz.so",
            "libc++_shared.so", "libandroid.so"}


def run(*args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def needed(path):
    out = subprocess.run(["readelf", "-d", str(path)], check=True,
                         capture_output=True, text=True).stdout
    return [line.split("[")[1].rstrip("]")
            for line in out.splitlines() if "(NEEDED)" in line]


def main():
    for name in STAGED:
        if not (SOURCE / name).is_file():
            raise SystemExit(f"Linux binary missing: {name}. Run scripts/fetch-linux.sh first.")
    shutil.rmtree(DEST.parent, ignore_errors=True)
    DEST.mkdir(parents=True)
    for source, target in STAGED.items():
        shutil.copyfile(SOURCE / source, DEST / target)
        run("chmod", "755", DEST / target)
    for library in sorted(DEST.glob("*.so")):
        deps = needed(library)
        foreign = [d for d in deps if d not in PLATFORM]
        if foreign:
            raise SystemExit(f"{library.name} needs non-platform libraries: {foreign}")
    manifest = {"libraries": sorted(STAGED.values()),
                "binaries": STAGED}
    (DEST.parent / "linuxjni.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Linux runtime staged: {len(STAGED)} executables, "
          f"{sum(p.stat().st_size for p in DEST.glob('*.so')) // 1024} KiB")


if __name__ == "__main__":
    main()
