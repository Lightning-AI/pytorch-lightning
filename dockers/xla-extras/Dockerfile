FROM google/cloud-sdk:slim

# CALL: docker image build -t pytorch-lightning:XLA-extras-py3.6 -f dockers/xla-extras/Dockerfile .
# This Dockerfile installs pytorch/xla 3.7 wheels. There are also 3.6 wheels available; see below.
ARG PYTHON_VERSION=3.6

SHELL ["/bin/bash", "-c"]

# for skipping configurations
ENV DEBIAN_FRONTEND=noninteractive

# show system inforation
RUN lsb_release -a && cat /etc/*-release

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        # python${PYTHON_VERSION}-dev \
        # python${PYTHON_VERSION}-distutils \
        build-essential \
        cmake \
        wget \
        unzip \
        ca-certificates \
        libomp5 \
    && \

# Install python dependencies
    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll --no-check-certificate && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py && \

# Set the default python and install PIP packages
    update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \

# Disable cache
    pip config set global.cache-dir false && \
    pip install "pip>20.1" -U  && \

# Install Pytorch XLA
    py_version=${PYTHON_VERSION/./} && \
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
    pip install mkl && \

# Cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache

# Install pytorch-lightning dependencies.
RUN \
# Get package
    wget https://github.com/PyTorchLightning/pytorch-lightning/archive/master.zip --progress=bar:force:noscroll && \
    unzip master.zip && \
    rm master.zip && \

# Install PL dependencies
    cd pytorch-lightning-master && \
    pip install --requirement ./requirements/base.txt && \
    # Drop Horovod
    python -c "fname = 'requirements/extra.txt' ; lines = [line for line in open(fname).readlines() if not line.startswith('horovod')] ; open(fname, 'w').writelines(lines)" && \
    pip install --requirement ./requirements/extra.txt && \
    cd .. && \
    rm -rf pytorch-lightning-master && \
    rm -rf /root/.cache

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/"

RUN python -c "import torch; print(torch.__version__)"
