ARG CUDA_VERSION=10.1
FROM nvidia/cuda:${CUDA_VERSION}-devel

# install versions
ARG PYTHON_VERSION=3.7
ARG PYTORCH_VERSION=1.6
ARG PYTORCH_CHANNEL=pytorch
ARG LIGHTNING_VERSION=""
# NOTE new Conda does not forward the exit status... https://github.com/conda/conda/issues/8385
ARG CONDA_VERSION=4.7.12

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
    && \

# Cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache
         
# add non-root user
RUN useradd --create-home --shell /bin/bash flash

USER flash
ENV CONDA_ENV=lightning
ENV WORKDIR=/home/flash
WORKDIR $WORKDIR

# install conda and python
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ${WORKDIR}/miniconda && \
    rm ~/miniconda.sh

# add conda to path
ENV PATH="${WORKDIR}/miniconda/bin:$PATH"
ENV LD_LIBRARY_PATH="${WORKDIR}/miniconda/lib:$LD_LIBRARY_PATH"
ENV CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda"

COPY --chown=flash environment.yml environment.yml

# conda init
RUN conda create -y --name $CONDA_ENV python=$PYTHON_VERSION pytorch=$PYTORCH_VERSION torchvision cudatoolkit=$CUDA_VERSION --channel=$PYTORCH_CHANNEL && \
    conda init bash && \
    # NOTE: this requires that the channel is presented in the yaml before packages
    python -c "fname = 'environment.yml' ; req = open(fname).read().replace('pytorch', '${PYTORCH_CHANNEL}', 1) ; open(fname, 'w').write(req)" && \
    conda env update --file environment.yml && \
    conda clean -ya && \
    rm environment.yml

ENV LD_LIBRARY_PATH="${WORKDIR}/miniconda/envs/${CONDA_ENV}/lib:$LD_LIBRARY_PATH"
# if you want this environment to be the default one, uncomment the following line:
ENV CONDA_DEFAULT_ENV=${CONDA_ENV}

# Get package
COPY --chown=flash ./ ./pytorch-lightning/

# install dependencies
RUN \
    # Disable cache
    #conda install "pip>20.1" && \
    #pip config set global.cache-dir false && \
    if [ -z $LIGHTNING_VERSION ] ; then \
        pip install ./pytorch-lightning --upgrade-strategy only-if-needed ; \
        rm -rf pytorch-lightning ; \
    else \
        rm -rf pytorch-lightning ; \
        pip install git+https://github.com/PyTorchLightning/pytorch-lightning.git@${LIGHTNING_VERSION} --upgrade-strategy only-if-needed ; \
    fi

RUN python --version && \
    pip --version && \
    pip list && \
    python -c "import pytorch_lightning as pl; print(pl.__version__)"
        
CMD ["/bin/bash"]
