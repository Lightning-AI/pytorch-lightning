# Existing images:
# --build-arg TORCH_VERSION=1.6.0 --build-arg CUDA_VERSION=10.1
# --build-arg TORCH_VERSION=1.5 --build-arg CUDA_VERSION=10.1
# --build-arg TORCH_VERSION=1.4 --build-arg CUDA_VERSION=10.1
# --build-arg TORCH_VERSION=1.3 --build-arg CUDA_VERSION=10.1
# --build-arg TORCH_VERSION=1.2 --build-arg CUDA_VERSION=10.0
# --build-arg TORCH_VERSION=1.1.0 --build-arg CUDA_VERSION=10.0 --build-arg CUDNN_VERSION=7.5

ARG CUDNN_VERSION=7
ARG CUDA_VERSION=10.1

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel

ARG PYTHON_VERSION
ARG PYTORCH_VERSION
ARG PYTORCH_CHANNEL
ARG CONDA_VERSION=4.7.12

SHELL ["/bin/bash", "-c"]

ENV HOROVOD_GPU_ALLREDUCE=NCCL
ENV HOROVOD_GPU_BROADCAST=NCCL
ENV HOROVOD_WITH_PYTORCH=1
ENV HOROVOD_WITHOUT_TENSORFLOW=1
ENV HOROVOD_WITHOUT_MXNET=1
ENV HOROVOD_WITH_GLOO=1
ENV HOROVOD_WITHOUT_MPI=1
ENV PATH="$PATH:/root/.local/bin"
ENV MAKEFLAGS="-j$(nproc)"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./tests/install_AMP.sh install_AMP.sh
COPY ./requirements/base.txt requirements.txt
COPY ./requirements/extra.txt requirements-extra.txt
COPY ./requirements/test.txt requirements-tests.txt
COPY ./requirements/examples.txt requirements-examples.txt

# install conda and python
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /root/conda && \
    rm ~/miniconda.sh && \
    /root/conda/bin/conda install -y python=$PYTHON_VERSION && \
    /root/conda/bin/conda install -y pytorch=$PYTORCH_VERSION torchvision cudatoolkit=$CUDA_VERSION -c $PYTORCH_CHANNEL && \
    /root/conda/bin/conda clean -ya

# add conda to path
ENV PATH /root/conda/bin:$PATH

# install dependencies
RUN \
    # Install AMP
    bash install_AMP.sh && \
    # Install all requirements
    pip install -r requirements.txt --upgrade-strategy only-if-needed && \
    # HOROVOD_BUILD_ARCH_FLAGS="-mfma" && \
    pip install -r requirements-extra.txt && \
    # pip install -r requirements-examples.txt && \
    pip install -r requirements-tests.txt && \
    rm install_AMP.sh && \
    rm requirements* && \
    # Cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    # Show what we have
    pip --version && \
    pip list
