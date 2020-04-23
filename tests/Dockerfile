ARG TORCH_VERSION=1.4
ARG CUDA_VERSION=10.1

FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn7-devel

ENV HOROVOD_GPU_ALLREDUCE: NCCL
ENV HOROVOD_GPU_BROADCAST: NCCL
ENV HOROVOD_WITH_PYTORCH: 1
ENV HOROVOD_WITHOUT_TENSORFLOW: 1
ENV HOROVOD_WITHOUT_MXNET: 1
ENV HOROVOD_WITH_GLOO: 1
ENV HOROVOD_WITHOUT_MPI: 1
ENV PATH: "$PATH:/root/.local/bin"
ENV MAKEFLAGS: "-j$(nproc)"

COPY ./tests/install_AMP.sh install_AMP.sh
COPY ./requirements.txt requirements.txt
COPY ./requirements-extra.txt requirements-extra.txt
COPY ./tests/requirements.txt requirements-tests.txt

# Install AMP
RUN apt-get update && apt-get install -y cmake && \
    bash install_AMP.sh && \
    pip install -r requirements.txt --user && \
    pip install -r requirements-extra.txt --user && \
    pip install -r requirements-tests.txt --user && \
    pip list
