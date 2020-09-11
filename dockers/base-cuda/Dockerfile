# Existing images:
# --build-arg PYTHON_VERSION=3.7 --build-arg PYTORCH_VERSION=1.7 --build-arg PYTORCH_CHANNEL=pytorch-nightly --build-arg CUDA_VERSION=10.1
# --build-arg PYTHON_VERSION=3.7 --build-arg PYTORCH_VERSION=1.6 --build-arg PYTORCH_CHANNEL=pytorch --build-arg CUDA_VERSION=10.1
# --build-arg PYTHON_VERSION=3.7 --build-arg PYTORCH_VERSION=1.5 --build-arg PYTORCH_CHANNEL=pytorch --build-arg CUDA_VERSION=10.1
# --build-arg PYTHON_VERSION=3.7 --build-arg PYTORCH_VERSION=1.4 --build-arg PYTORCH_CHANNEL=pytorch --build-arg CUDA_VERSION=10.1
# --build-arg PYTHON_VERSION=3.7 --build-arg PYTORCH_VERSION=1.3 --build-arg PYTORCH_CHANNEL=pytorch --build-arg CUDA_VERSION=10.1

ARG CUDNN_VERSION=7
ARG CUDA_VERSION=10.1

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel

ARG PYTHON_VERSION=3.7
ARG PYTORCH_VERSION=1.6
ARG PYTORCH_CHANNEL=pytorch
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
# TODO: uncomment in horovod next release, https://github.com/horovod/horovod/pull/2239
# ENV MAKEFLAGS="-j$(nproc)"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    && \
    # Cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*

# add non-root user
RUN useradd --create-home --shell /bin/bash flash

USER flash
ENV CONDA_ENV=lightning
ENV WORKDIR=/home/flash
WORKDIR $WORKDIR

COPY --chown=flash environment.yml environment.yml

# install conda and python
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ${WORKDIR}/miniconda && \
    rm ~/miniconda.sh

# add conda to path
ENV PATH="${WORKDIR}/miniconda/bin:$PATH"
ENV LD_LIBRARY_PATH="${WORKDIR}/miniconda/lib:$LD_LIBRARY_PATH"
ENV CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda"

# conda init
RUN conda create -y --name $CONDA_ENV python=$PYTHON_VERSION pytorch=$PYTORCH_VERSION cudatoolkit=$CUDA_VERSION --channel=$PYTORCH_CHANNEL && \
    conda init bash && \
    # NOTE: this requires that the channel is presented in the yaml before packages
    python -c "fname = 'environment.yml' ; req = open(fname).read().replace('pytorch', '${PYTORCH_CHANNEL}', 1) ; open(fname, 'w').write(req)" && \
    conda env update --file environment.yml && \
    conda clean -ya && \
    rm environment.yml && \
    # Disable cache
    conda install "pip>20.1" -y && \
    pip config set global.cache-dir false

ENV PATH ${WORKDIR}/miniconda/envs/${CONDA_ENV}/bin:$PATH
ENV LD_LIBRARY_PATH="${WORKDIR}/miniconda/envs/${CONDA_ENV}/lib:$LD_LIBRARY_PATH"
# if you want this environment to be the default one, uncomment the following line:
ENV CONDA_DEFAULT_ENV=${CONDA_ENV}

COPY ./requirements/test.txt requirements-tests.txt
COPY ./requirements/examples.txt requirements-examples.txt

RUN \
    echo ". ${WORKDIR}/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ${CONDA_ENV}" >> ~/.bashrc && \
    source ~/.bashrc && \
    # Install all requirements
    pip install -r requirements-tests.txt --upgrade-strategy only-if-needed && \
    pip install -r requirements-examples.txt --upgrade-strategy only-if-needed && \
    rm requirements* && \
    # Show what we have
    pip --version && \
    conda info && \
    conda list && \
    pip list

CMD ["bin/bash"]
