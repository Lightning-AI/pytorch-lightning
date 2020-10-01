# Existing images:
# --build-arg PYTHON_VERSION=3.7 --build-arg PYTORCH_VERSION=1.7 --build-arg PYTORCH_CHANNEL=pytorch-nightly --build-arg CUDA_VERSION=10.1
# --build-arg PYTHON_VERSION=3.7 --build-arg PYTORCH_VERSION=1.6 --build-arg PYTORCH_CHANNEL=pytorch --build-arg CUDA_VERSION=10.1
# --build-arg PYTHON_VERSION=3.7 --build-arg PYTORCH_VERSION=1.5 --build-arg PYTORCH_CHANNEL=pytorch --build-arg CUDA_VERSION=10.1
# --build-arg PYTHON_VERSION=3.7 --build-arg PYTORCH_VERSION=1.4 --build-arg PYTORCH_CHANNEL=pytorch --build-arg CUDA_VERSION=10.1
# --build-arg PYTHON_VERSION=3.7 --build-arg PYTORCH_VERSION=1.3 --build-arg PYTORCH_CHANNEL=pytorch --build-arg CUDA_VERSION=10.1

ARG CUDNN_VERSION=7
ARG CUDA_VERSION=10.1

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel
# FROM nvidia/cuda:${CUDA_VERSION}-devel

ARG PYTHON_VERSION=3.7
ARG PYTORCH_VERSION=1.6
ARG PYTORCH_CHANNEL=pytorch
ARG CONDA_VERSION=4.7.12

SHELL ["/bin/bash", "-c"]

ENV PATH="$PATH:/root/.local/bin"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
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
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:$PATH"
ENV LD_LIBRARY_PATH="/root/miniconda3/lib:$LD_LIBRARY_PATH"
ENV CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda"

ENV HOROVOD_GPU_OPERATIONS=NCCL
ENV HOROVOD_WITH_PYTORCH=1
ENV HOROVOD_WITHOUT_TENSORFLOW=1
ENV HOROVOD_WITHOUT_MXNET=1
ENV HOROVOD_WITH_GLOO=1
ENV HOROVOD_WITHOUT_MPI=1

ENV CONDA_ENV=lightning
COPY environment.yml environment.yml

# conda init
RUN conda create -y --name $CONDA_ENV "python=$PYTHON_VERSION" "pytorch=$PYTORCH_VERSION" torchvision "cudatoolkit=$CUDA_VERSION" --channel=$PYTORCH_CHANNEL && \
    conda init bash && \
    # NOTE: this requires that the channel is presented in the yaml before packages
    # replace channel to nigtly if neede, fix PT version and remove Horovod as it will be installe later
    python -c "import re ; fname = 'environment.yml' ; req = re.sub(r'torch>=[\d\.]+', 'torch=${PYTORCH_VERSION}', open(fname).read().replace('pytorch', '${PYTORCH_CHANNEL}', 1)) ; open(fname, 'w').writelines([l for l in req if 'horovod' not in l])" && \
    cat environment.yml && \
    conda env update --file environment.yml && \
    conda clean -ya && \
    # conda config --set auto_activate_base true && \
    rm environment.yml

ENV PATH /root/miniconda3/envs/${CONDA_ENV}/bin:$PATH
ENV LD_LIBRARY_PATH="/root/miniconda3/envs/${CONDA_ENV}/lib:$LD_LIBRARY_PATH"
# if you want this environment to be the default one, uncomment the following line:
ENV CONDA_DEFAULT_ENV=${CONDA_ENV}

COPY ./requirements/extra.txt requirements-extra.txt
COPY ./requirements/test.txt requirements-tests.txt
COPY ./requirements/examples.txt requirements-examples.txt

RUN \
    # Disable cache
    pip config set global.cache-dir false && \
    #echo ". ${WORKDIR}/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    #echo "conda activate ${CONDA_ENV}" >> ~/.bashrc && \
    #source ~/.bashrc && \
    # filter only Horovod
    python -c "fname = 'requirements-extra.txt' ; req = open(fname).readlines() ; open(fname, 'w').writelines([l for l in req if 'horovod' in l])" && \
    # Install all requirements
    MAKEFLAGS="-j$(nproc)" ; pip install -r requirements-extra.txt && \
    pip install -r requirements-tests.txt --upgrade-strategy only-if-needed && \
    pip install -r requirements-examples.txt --upgrade-strategy only-if-needed && \
    rm requirements*

RUN \
    # Show what we have
    pip --version && \
    conda info && \
    conda list && \
    pip list && \
    python -c "import torch; print(torch.__version__)"