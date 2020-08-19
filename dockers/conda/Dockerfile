ARG CUDA_VERSION=10.1
FROM nvidia/cuda:${CUDA_VERSION}-base

# install versions
ARG PYTHON_VERSION=3.7
ARG PYTORCH_VERSION=1.4
ARG LIGHTNING_VERSION=master
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
RUN useradd --create-home --shell /bin/bash containeruser

USER containeruser
WORKDIR /home/containeruser
ENV CONDA_ENV=lightning

# install conda and python
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /home/containeruser/miniconda && \
    rm ~/miniconda.sh

# add conda to path
ENV PATH="/home/containeruser/miniconda/bin:$PATH"
ENV LD_LIBRARY_PATH="/home/containeruser/miniconda/lib:$LD_LIBRARY_PATH"

# conda init
RUN conda create -y --name $CONDA_ENV python=$PYTHON_VERSION && \
    conda init bash && \
    conda install -y python=$PYTHON_VERSION && \

# Disable cache
    pip config set global.cache-dir false && \
    pip install "pip>20.1" -U

ENV LD_LIBRARY_PATH="/home/containeruser/miniconda/envs/$CONDA_ENV/lib:$LD_LIBRARY_PATH"
# if you want this environment to be the default one, uncomment the following line:
ENV CONDA_DEFAULT_ENV=${CONDA_ENV}

# install dependencies
RUN pip install torch==$PYTORCH_VERSION  -f https://download.pytorch.org/whl/torch_stable.html && \
    git clone https://github.com/PyTorchLightning/pytorch-lightning.git --single-branch --branch $LIGHTNING_VERSION && \
    pip install ./pytorch-lightning && \
    pip install -r pytorch-lightning/requirements/extra.txt && \
    rm -rf pytorch-lightning

RUN python --version && \
    pip --version && \
    python -c "import pytorch_lightning as pl; print(pl.__version__)"
        
CMD ["/bin/bash"]
