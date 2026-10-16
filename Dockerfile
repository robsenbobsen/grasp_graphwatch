# Single image shared by every doc-comp_* detection service in docker-compose.yml:
# the REST API (detection_api), GRASP CLIs (main.py, train.py, infer.py), the
# contextualization workbench and the sysdig data pipeline.
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

# sysdig CLI, only needed to read .scap files (graphwatch_data_pipeline/sysdig). Debian's
# package (0.29) cannot read captures of current sysdig versions, so take the binary of the
# official release; reading captures needs neither its kernel driver nor dkms.
ARG SYSDIG_VERSION=0.41.4
ADD --checksum=sha256:efbd2ac6ebb155d68bc2a2509d02b70d6496ba6d0f11b6e9f0c9f3bd6c41f89d \
    https://github.com/draios/sysdig/releases/download/${SYSDIG_VERSION}/sysdig-${SYSDIG_VERSION}-x86_64.deb \
    /tmp/sysdig.deb
RUN dpkg-deb -x /tmp/sysdig.deb /tmp/sysdig \
    && install -m 0755 /tmp/sysdig/usr/bin/sysdig /usr/local/bin/sysdig \
    && rm -rf /tmp/sysdig /tmp/sysdig.deb

WORKDIR /detection_service

RUN pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
    && pip install torch_geometric \
    && pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
        -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${TORCH_VARIANT}.html

COPY requirements.txt ./requirements.txt
COPY grasp_contextualization/requirements.txt ./grasp_contextualization/requirements.txt
COPY detection_api/requirements.txt ./detection_api/requirements.txt
RUN pip install -r requirements.txt -r grasp_contextualization/requirements.txt \
        -r detection_api/requirements.txt

COPY . .

CMD ["uvicorn", "detection_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
