# Single image shared by every doc-comp_* detection service in docker-compose.yml:
# GRASP (main.py), the contextualization workbench and the sysdig data pipeline.
#
# Built for NVIDIA GPUs (CUDA 12.9 wheels, which bundle the CUDA runtime; the host
# needs the NVIDIA driver + NVIDIA Container Toolkit). TORCH_VARIANT=cpu builds a
# CPU-only image instead.
FROM python:3.12.7-slim-bookworm

ARG TORCH_VERSION=2.8.0
ARG TORCH_VARIANT=cu129

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Containers run as the host UID (see docker-compose.yml), which has no passwd entry,
    # so HOME would be the unwritable "/". Caches/config of torch, streamlit and
    # matplotlib go to $HOME, so point it somewhere writable.
    HOME=/tmp \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# sysdig CLI is only needed to read .scap files (graphwatch_data_pipeline/sysdig);
# reading captures does not need the kernel module, so skip recommends (dkms).
RUN apt-get update \
    && apt-get install -y --no-install-recommends sysdig

WORKDIR /detection_service

RUN pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
    && pip install torch_geometric \
    && pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
        -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${TORCH_VARIANT}.html

COPY requirements.txt ./requirements.txt
COPY grasp_contextualization/requirements.txt ./grasp_contextualization/requirements.txt
RUN pip install -r requirements.txt -r grasp_contextualization/requirements.txt

COPY . .

CMD ["python", "main.py", "--experiment-config", "grasp/experiments/all_experiments/cadets_e3_default/experiment_cadets_e3.yaml"]
