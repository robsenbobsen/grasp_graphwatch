#!/usr/bin/env bash
# Installs the NVIDIA Container Toolkit on Debian/Ubuntu and enables it for Docker,
# so the doc-comp_* containers can use the GPUs. Run with: sudo ./scripts/install_nvidia_container_toolkit.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run as root: sudo $0" >&2
  exit 1
fi

KEYRING=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o "$KEYRING"
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed "s#deb https://#deb [signed-by=$KEYRING] https://#g" \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit

nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "Done. Check with: docker run --rm --gpus all doc-comp_grasp:latest nvidia-smi"
