ARG PYTHON_VERSION=3.7
ARG PYTORCH_VERSION=1.5

FROM pytorchlightning/pytorch_lightning:base-cuda-py${PYTHON_VERSION}-torch${PYTORCH_VERSION}

ARG LIGHTNING_VERSION=""

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
