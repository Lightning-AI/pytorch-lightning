FROM google/cloud-sdk:slim

# Build args.
ARG GITHUB_REF=refs/heads/master
ARG TEST_IMAGE=0

# TODO: This Dockerfile installs pytorch/xla 3.6 wheels. There are also 3.7
# wheels available; see below.
ENV PYTHON_VERSION=3.6

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates

# Install conda and python.
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b && \
    rm ~/miniconda.sh

ENV PATH=/root/miniconda3/bin:$PATH

RUN conda create -y --name container python=$PYTHON_VERSION

# Run the rest of commands within the new conda env.
SHELL ["conda", "run", "-n", "container", "/bin/bash", "-c"]
RUN conda install -y python=$PYTHON_VERSION mkl

RUN pip uninstall -y torch
# TODO: Python 3.7 wheels are available. Replace cp36-cp36m with cp37-cp37m.
RUN gsutil cp 'gs://tpu-pytorch/wheels/torch-nightly-cp36-cp36m-linux_x86_64.whl' .
RUN gsutil cp 'gs://tpu-pytorch/wheels/torch_xla-nightly-cp36-cp36m-linux_x86_64.whl' .
RUN gsutil cp 'gs://tpu-pytorch/wheels/torchvision-nightly-cp36-cp36m-linux_x86_64.whl' .
RUN pip install 'torch-nightly-cp36-cp36m-linux_x86_64.whl'
RUN pip install 'torch_xla-nightly-cp36-cp36m-linux_x86_64.whl'
RUN pip install 'torchvision-nightly-cp36-cp36m-linux_x86_64.whl'
RUN rm 'torch-nightly-cp36-cp36m-linux_x86_64.whl'
RUN rm 'torch_xla-nightly-cp36-cp36m-linux_x86_64.whl'
RUN rm 'torchvision-nightly-cp36-cp36m-linux_x86_64.whl'
RUN apt-get install -y libomp5

ENV LD_LIBRARY_PATH=root/miniconda3/envs/container/lib

# Install pytorch-lightning at the current PR, plus dependencies.
RUN git clone https://github.com/PyTorchLightning/pytorch-lightning.git && \
    cd pytorch-lightning && \
    git fetch origin $GITHUB_REF:CI && \
    git checkout CI && \
    cd .. && \
    pip install ./pytorch-lightning

# If using this image for tests, intall more dependencies and don't delete
# the source code where the tests live.
RUN if [ $TEST_IMAGE -eq 1 ] ; then pip install -r pytorch-lightning/requirements/test.txt ; else rm -rf pytorch-lightning ; fi

RUN python -c "import pytorch_lightning as pl; print(pl.__version__)"
RUN conda init bash
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
