FROM google/cloud-sdk:slim

# CALL: docker image build -t pytorch-lightning:XLA-extras-py3.6 -f dockers/base-xla/Dockerfile .
# This Dockerfile installs pytorch/xla 3.7 wheels. There are also 3.6 wheels available; see below.
ARG PYTHON_VERSION=3.7
ARG XLA_VERSION="1.7"

SHELL ["/bin/bash", "-c"]

# for skipping configurations
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_ENV=pytorch-xla

# show system inforation
RUN lsb_release -a && cat /etc/*-release

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        wget \
        curl \
        unzip \
        ca-certificates \
        libomp5 \
    && \

# Install conda and python.
# NOTE new Conda does not forward the exit status... https://github.com/conda/conda/issues/8385
    curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b && \
    rm ~/miniconda.sh && \

# Cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache

ENV PATH="/root/miniconda3/bin:$PATH"
ENV LD_LIBRARY_PATH="/root/miniconda3/lib:$LD_LIBRARY_PATH"

RUN conda create -y --name $CONDA_ENV python=$PYTHON_VERSION && \
    conda init bash && \
    conda install -y python=$PYTHON_VERSION mkl && \

# Disable cache
    pip config set global.cache-dir false && \
    pip install "pip>20.1" -U  && \

# Install Pytorch XLA
    py_version=${PYTHON_VERSION/./} && \
    # Python 3.7 wheels are available. Replace cp36-cp36m with cp37-cp37m
    gsutil cp "gs://tpu-pytorch/wheels/torch-${XLA_VERSION}-cp${py_version}-cp${py_version}m-linux_x86_64.whl" . && \
    gsutil cp "gs://tpu-pytorch/wheels/torch_xla-${XLA_VERSION}-cp${py_version}-cp${py_version}m-linux_x86_64.whl" . && \
    gsutil cp "gs://tpu-pytorch/wheels/torchvision-${XLA_VERSION}-cp${py_version}-cp${py_version}m-linux_x86_64.whl" . && \
    pip install *.whl && \
    rm *.whl

ENV LD_LIBRARY_PATH="/root/miniconda3/envs/$CONDA_ENV/lib:$LD_LIBRARY_PATH"
# if you want this environment to be the default one, uncomment the following line:
ENV CONDA_DEFAULT_ENV=${CONDA_ENV}

# Install pytorch-lightning dependencies.
RUN \
# Get package
    wget https://github.com/PyTorchLightning/pytorch-lightning/archive/master.zip --progress=bar:force:noscroll && \
    unzip -q master.zip && \
    rm master.zip && \

# Install PL dependencies
    cd pytorch-lightning-master && \
    # drop Torch
    python -c "fname = \"./requirements/base.txt\" ; lines = [line for line in open(fname).readlines() if not line.startswith(\"torch\")] ; open(fname, \"w\").writelines(lines)" && \
    pip install --requirement ./requirements/base.txt && \
    # drop Horovod
    python -c "fname = \"./requirements/extra.txt\" ; lines = [line for line in open(fname).readlines() if not line.startswith(\"horovod\")] ; open(fname, \"w\").writelines(lines)" && \
    pip install --requirement ./requirements/extra.txt && \
    # drop TorchVision
    python -c "fname = \"./requirements/examples.txt\" ; lines = [line for line in open(fname).readlines() if not line.startswith(\"torchvision\")] ; open(fname, \"w\").writelines(lines)" && \
    pip install --requirement ./requirements/examples.txt && \
    cd .. && \
    rm -rf pytorch-lightning-master && \
    rm -rf /root/.cache

RUN pip --version && \
    python -c "import torch; print(torch.__version__)"
