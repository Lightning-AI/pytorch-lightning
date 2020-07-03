FROM google/cloud-sdk:slim

# Build args.
ARG GITHUB_REF=refs/heads/master
ARG TEST_IMAGE=0

# This Dockerfile installs pytorch/xla 3.7 wheels. There are also 3.6 wheels available; see below.
ARG PYTHON_VERSION=3.7

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        $( [ ${PYTHON_VERSION%%.*} -ge 3 ] && echo "python${PYTHON_VERSION%%.*}-distutils" ) \
        build-essential \
        cmake \
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
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \

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
    pip install mkl

# Install pytorch-lightning at the current PR, plus dependencies.
RUN wget https://github.com/PyTorchLightning/pytorch-lightning/archive/master.zip && \
    unzip master.zip && \
    cd pytorch-lightning && \
    pip install --requirement ./requirements/base.txt && \
    python -c "fname = 'requirements/extra.txt' ; lines = [line for line in open(fname).readlines() if not line.startswith('horovod')] ; open(fname, 'w').writelines(lines)" && \
    pip install --requirement ./requirements/extra.txt && \
    cd ..  && \
    rm -rf pytorch-lightning

RUN  apt-get -y remove \
        curl \
        wget \
    && \
    apt-get autoremove -y && \
    apt-get clean

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/"

RUN python -c "import torch; print(torch.__version__)"
