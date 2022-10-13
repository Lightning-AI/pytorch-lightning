# Run command to build:
#    gaudi_ver=$(curl -s "https://vault.habana.ai/artifactory/gaudi-docker/" |  sed -n 's/.*href="\([^"]*\).*/\1/p' | tail -2 | head -1 | sed "s/\///1")
#    pytorch_install_ver=$(curl -s "https://vault.habana.ai/artifactory/gaudi-docker/$gaudi_ver/ubuntu20.04/habanalabs/" | sed -n 's/.*href="\([^"]*\).*/\1/p'| sed "s/\///1" | grep pytorch-installer)
#    pytorch_install_ver=${pytorch_install_ver/"pytorch-installer-"/""}
#    docker build -t gaudi-docker-agent:latest \
#       --build-arg GAUDI_VERSION=$gaudi_ver \
#       --build-arg PYTORCH_INSTALLER_VERSION=$pytorch_install_ver \
#       -f Dockerfile .
# Run command:
#    docker run --privileged \
#        -v /dev:/dev \
#        -e AZP_URL="https://dev.azure.com/ORGANIZATION/" \
#        -e AZP_TOKEN="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" \
#        -e AZP_AGENT_NAME="hpu1" \
#        -e AZP_POOL="intel-hpus" \
#        gaudi-docker-agent:latest

ARG DIST="latest"
ARG GAUDI_VERSION="1.6.1"
ARG PYTORCH_INSTALLER_VERSION="1.12.0"
FROM vault.habana.ai/gaudi-docker/${GAUDI_VERSION}/ubuntu20.04/habanalabs/pytorch-installer-${PYTORCH_INSTALLER_VERSION}:${DIST}

LABEL maintainer="https://vault.habana.ai/"
# update the base packages and add a non-sudo user
RUN \
    apt-get update -y && \
    apt-get upgrade -y && \
    useradd -m docker

# To make it easier for build and release pipelines to run apt-get,
# configure apt to not require confirmation (assume the -y argument by default)
ENV DEBIAN_FRONTEND=noninteractive
RUN echo "APT::Get::Assume-Yes \"true\";" > /etc/apt/apt.conf.d/90assumeyes

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        curl \
        jq \
        git \
        iputils-ping \
        libcurl4 \
        libunwind8 \
        netcat \
        libssl1.0 \
        libssl-dev \
        libffi-dev \
        python3 \
        python3-venv \
        python3-dev \
        python3-pip

RUN curl -fsSL https://get.docker.com -o get-docker.sh && \
    sh get-docker.sh && \
    rm get-docker.sh

RUN pip uninstall pytorch-lightning -y

WORKDIR /azp

COPY ./.azure/start.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/start.sh

ENTRYPOINT ["/usr/local/bin/start.sh"]
