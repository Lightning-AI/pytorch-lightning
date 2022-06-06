ARG DIST="latest"
ARG VERSION="1.4.0"
ARG PYTORCH_INSTALLER_VERSION="pytorch-installer-1.10.2"

FROM vault.habana.ai/gaudi-docker/${VERSION}/ubuntu20.04/habanalabs/${PYTORCH_INSTALLER_VERSION}:${DIST}

LABEL maintainer="https://vault.habana.ai/"

RUN echo "ALL ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

WORKDIR /azp

COPY ./dockers/ci-runner-hpu/start.sh /usr/local/bin/

RUN chmod +x /usr/local/bin/start.sh

ENTRYPOINT ["/usr/local/bin/start.sh"]
CMD ["bash"]
