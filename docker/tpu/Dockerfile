FROM google/cloud-sdk:slim

# Build args.
ARG GITHUB_REF=refs/heads/master
ARG TEST_IMAGE=0

# This Dockerfile installs pytorch/xla 3.7 wheels. There are also 3.6 wheels available; see below.
ENV PYTHON_VERSION=3.7

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        $( [ ${PYTHON_VERSION%%.*} -ge 3 ] && echo "python${PYTHON_VERSION%%.*}-distutils" ) \
        build-essential \
        cmake \
        git \
        curl \
        wget \
        ca-certificates \
        libomp5 \
    && \

# install python dependencies
    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll --no-check-certificate && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py && \

# Set the default python and install PIP packages
    update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

RUN py_version=${PYTHON_VERSION/./} && \
    # Python 3.7 wheels are available. Replace cp36-cp36m with cp37-cp37m
    gsutil cp "gs://tpu-pytorch/wheels/torch-nightly-cp${py_version}-cp${py_version}m-linux_x86_64.whl" . && \
    gsutil cp "gs://tpu-pytorch/wheels/torch_xla-nightly-cp${py_version}-cp${py_version}m-linux_x86_64.whl" . && \
    gsutil cp "gs://tpu-pytorch/wheels/torchvision-nightly-cp${py_version}-cp${py_version}m-linux_x86_64.whl" . && \
    pip install "torch-nightly-cp${py_version}-cp${py_version}m-linux_x86_64.whl" && \
    pip install "torch_xla-nightly-cp${py_version}-cp${py_version}m-linux_x86_64.whl" && \
    pip install "torchvision-nightly-cp${py_version}-cp${py_version}m-linux_x86_64.whl" && \
    rm "torch-nightly-cp${py_version}-cp${py_version}m-linux_x86_64.whl" && \
    rm "torch_xla-nightly-cp${py_version}-cp${py_version}m-linux_x86_64.whl" && \
    rm "torchvision-nightly-cp${py_version}-cp${py_version}m-linux_x86_64.whl" && \
    pip install mkl

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/"

RUN python -c "import torch; print(torch.__version__)"

# Install pytorch-lightning at the current PR, plus dependencies.
RUN git clone https://github.com/PyTorchLightning/pytorch-lightning.git && \
    cd pytorch-lightning && \
    git fetch origin $GITHUB_REF:CI && \
    git checkout CI && \
    pip install --requirement ./requirements/base.txt

# If using this image for tests, intall more dependencies and don"t delete
# the source code where the tests live.
RUN \
    # TODO: use conda sources if possible
    # drop Horovod
    # python -c "fname = './pytorch-lightning/requirements/extra.txt' ; lines = [ln for ln in open(fname).readlines() if not ln.startswith('horovod')] ; open(fname, 'w').writelines(lines)" && \
    # pip install -r pytorch-lightning/requirements/extra.txt ; && \
    if [ $TEST_IMAGE -eq 1 ] ; then \
        pip install -r pytorch-lightning/requirements/test.txt ; \
    else \
        rm -rf pytorch-lightning ; \
    fi

#RUN python -c "import pytorch_lightning as pl; print(pl.__version__)"

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
