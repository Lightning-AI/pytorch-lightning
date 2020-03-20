ARG TORCH_VERSION=1.4
ARG CUDA_VERSION=10.1

FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn7-runtime

# Install AMP
RUN bash ./tests/install_AMP.sh
