ARG DIST="latest"
ARG GAUDI_VERSION="1.4.0"
ARG PYTORCH_VERSION="1.10.2"

FROM vault.habana.ai/gaudi-docker/${GAUDI_VERSION}/ubuntu20.04/habanalabs/pytorch-installer-${PYTORCH_VERSION}:${DIST}

LABEL maintainer="https://vault.habana.ai/"

RUN echo "ALL ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

WORKDIR /azp

COPY ./dockers/ci-runner-hpu/start.sh /usr/local/bin/

RUN chmod +x /usr/local/bin/start.sh

RUN curl -fsSL https://get.docker.com -o get-docker.sh && \
    sh get-docker.sh && \
    rm get-docker.sh

#RUN docker --help

ENTRYPOINT ["/usr/local/bin/start.sh"]
CMD ["bash"]
