ARG CUDA_VERSION=10.2
FROM nvidia/cuda:${CUDA_VERSION}-base

# install versions
ARG PYTHON_VERSION=3.8
ARG PYTORCH_VERSION=1.5
ARG LIGHTNING_VERSION=master

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates
         
# add non-root user
RUN useradd --create-home --shell /bin/bash containeruser
USER containeruser
WORKDIR /home/containeruser
         
   
# install conda and python
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /home/containeruser/conda && \
    rm ~/miniconda.sh && \
    /home/containeruser/conda/bin/conda clean -ya && \
    /home/containeruser/conda/bin/conda install -y python=$PYTHON_VERSION 
     
# add conda to path
ENV PATH /home/containeruser/conda/bin:$PATH

# install dependencies
RUN pip install torch==$PYTORCH_VERSION
RUN git clone https://github.com/PyTorchLightning/pytorch-lightning.git --single-branch --branch $LIGHTNING_VERSION && \
        pip install -r pytorch-lightning/requirements.txt && \
        pip install -r pytorch-lightning/requirements-extra.txt && \
        pip install ./pytorch-lightning
        
CMD ["/bin/bash"]
