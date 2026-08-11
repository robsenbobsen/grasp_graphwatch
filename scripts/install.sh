#!/usr/bin/env bash
# Sync the project with the right torch/PyG backend for this machine —
# detects a CUDA-capable GPU via nvidia-smi and picks a matching build,
# falling back to CPU-only otherwise. No manual --extra flag needed.
#
# Any extra arguments are forwarded to `uv sync` (e.g. `./scripts/install.sh
# --extra cu128` to override detection, or `--frozen` to skip re-locking).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

backend="$(python3 scripts/detect_torch_backend.py)"
echo "grasp: detected torch backend '${backend}'" >&2

exec uv sync --extra "${backend}" "$@"
