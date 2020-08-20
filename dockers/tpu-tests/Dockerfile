ARG PYTHON_VERSION=3.7
ARG PYTORCH_VERSION=1.6

FROM pytorchlightning/pytorch_lightning:base-xla-py${PYTHON_VERSION}-torch${PYTORCH_VERSION}

#SHELL ["/bin/bash", "-c"]

COPY ./ ./pytorch-lightning/

# If using this image for tests, intall more dependencies and don"t delete the source code where the tests live.
RUN \
    # Install pytorch-lightning at the current PR, plus dependencies.
    pip install --requirement pytorch-lightning/requirements/base.txt --no-cache-dir && \
    # drop Horovod
    #python -c "fname = 'pytorch-lightning/requirements/extra.txt' ; lines = [line for line in open(fname).readlines() if not line.startswith('horovod')] ; open(fname, 'w').writelines(lines)" && \
    #pip install --requirement pytorch-lightning/requirements/extra.txt --no-cache-dir && \
    pip install --requirement pytorch-lightning/requirements/test.txt --no-cache-dir

#RUN python -c "import pytorch_lightning as pl; print(pl.__version__)"

COPY ./dockers/tpu-tests/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
