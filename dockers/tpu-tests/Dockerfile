ARG PYTHON_VERSION=3.7
ARG PYTORCH_VERSION=1.6

FROM pytorchlightning/pytorch_lightning:base-xla-py${PYTHON_VERSION}-torch${PYTORCH_VERSION}

# Build args.
ARG GITHUB_REF=refs/heads/master
ARG TEST_IMAGE=0

#SHELL ["/bin/bash", "-c"]

COPY ./ ./pytorch-lightning/

RUN ls -lh ./pytorch-lightning

# If using this image for tests, intall more dependencies and don"t delete the source code where the tests live.
RUN \
    # Install pytorch-lightning at the current PR, plus dependencies.
    pip install --requirement pytorch-lightning/requirements/base.txt --no-cache-dir && \
    # drop Horovod
    #python -c "fname = 'pytorch-lightning/requirements/extra.txt' ; lines = [line for line in open(fname).readlines() if not line.startswith('horovod')] ; open(fname, 'w').writelines(lines)" && \
    #pip install --requirement pytorch-lightning/requirements/extra.txt --no-cache-dir && \
    if [ $TEST_IMAGE -eq 1 ] ; then \
        pip install --requirement pytorch-lightning/requirements/test.txt --no-cache-dir ; \
    else \
        rm -rf pytorch-lightning ; \
    fi

#RUN python -c "import pytorch_lightning as pl; print(pl.__version__)"

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
