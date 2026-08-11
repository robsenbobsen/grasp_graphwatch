"""Pick the torch-geometric extra (cpu/cu126/cu128/cu130) for this machine.

Used by scripts/install.sh so `./scripts/install.sh` "just works" without the
user having to know their CUDA version or pass an --extra flag by hand.
Intentionally stdlib-only: it has to run *before* the project is synced, so
it can't rely on any third-party package being installed yet.

Run standalone to see the choice: `python3 scripts/detect_torch_backend.py`.
"""

import re
import shutil
import subprocess

# CUDA builds pyproject.toml offers, newest first. A driver reporting
# "CUDA Version: X.Y" in `nvidia-smi` can run any toolkit <= X.Y (the driver
# is backwards compatible), so we pick the newest build it supports.
SUPPORTED_BUILDS = [(13, 0, "cu130"), (12, 8, "cu128"), (12, 6, "cu126")]


def detect() -> str:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return "cpu"

    try:
        result = subprocess.run(
            [nvidia_smi], capture_output=True, text=True, timeout=10, check=True
        )
    except (subprocess.SubprocessError, OSError):
        # nvidia-smi exists but errored (e.g. driver/kernel-module mismatch) —
        # don't guess, just fall back to a build that always installs.
        return "cpu"

    # Older drivers print "CUDA Version: X.Y"; drivers from ~2026 (e.g. 610.x)
    # renamed the header field to "CUDA UMD Version: X.Y" instead.
    match = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", result.stdout)
    if not match:
        return "cpu"

    driver_version = (int(match.group(1)), int(match.group(2)))
    for major, minor, extra in SUPPORTED_BUILDS:
        if driver_version >= (major, minor):
            return extra

    # A CUDA-capable driver too old for our oldest build: fall back to CPU
    # rather than install a build that won't load.
    return "cpu"


if __name__ == "__main__":
    print(detect())
