ARG TORCH_VERSION=1.4
ARG CUDA_VERSION=10.1

FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn7-runtime

# Install AMP
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
    pip install -v --no-cache-dir ./ && \
    rm -rf apex
