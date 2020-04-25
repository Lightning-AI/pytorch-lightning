# Existing images:
# --build-arg TORCH_VERSION=1.5 --build-arg CUDA_VERSION=10.1
# --build-arg TORCH_VERSION=1.4 --build-arg CUDA_VERSION=10.1
# --build-arg TORCH_VERSION=1.3 --build-arg CUDA_VERSION=10.1
# --build-arg TORCH_VERSION=1.2 --build-arg CUDA_VERSION=10.0
# --build-arg TORCH_VERSION=1.1.0 --build-arg CUDA_VERSION=10.0 --build-arg CUDNN_VERSION=7.5

ARG TORCH_VERSION=1.4
ARG CUDA_VERSION=10.1
ARG CUDNN_VERSION=7

FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel

ENV HOROVOD_GPU_ALLREDUCE NCCL
ENV HOROVOD_GPU_BROADCAST NCCL
ENV HOROVOD_WITH_PYTORCH 1
ENV HOROVOD_WITHOUT_TENSORFLOW 1
ENV HOROVOD_WITHOUT_MXNET 1
ENV HOROVOD_WITH_GLOO 1
ENV HOROVOD_WITHOUT_MPI 1
ENV PATH "$PATH:/root/.local/bin"
ENV MAKEFLAGS "-j$(nproc)"

COPY ./tests/install_AMP.sh install_AMP.sh
COPY ./requirements/base.txt requirements.txt
COPY ./requirements/extra.txt requirements-extra.txt
COPY ./requirements/test.txt requirements-tests.txt

RUN apt-get update && apt-get install -y cmake && \
    # Install AMP
    bash install_AMP.sh && \
    pip install -r requirements.txt --user && \
    pip install -r requirements-extra.txt --user && \
    pip install -r requirements-tests.txt --user && \
    rm requirements* && \
    pip list
