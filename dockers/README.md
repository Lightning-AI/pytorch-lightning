# Docker images

## Builds images form attached Dockerfiles

You can build it on your own, note it takes lots of time, be prepared.

```bash
git clone <git-repository>
docker image build -t pytorch-lightning:latest -f dockers/conda/Dockerfile .
```

or with specific arguments

```bash
git clone <git-repository>
docker image build \
    -t pytorch-lightning:base-cuda-py3.8-pt1.8 \
    -f dockers/base-cuda/Dockerfile \
    --build-arg PYTHON_VERSION=3.8 \
    --build-arg PYTORCH_VERSION=1.8 \
    .
```
or nightly version from Conda
```bash
git clone <git-repository>
docker image build \
    -t pytorch-lightning:base-conda-py3.8-pt1.9 \
    -f dockers/base-conda/Dockerfile \
    --build-arg PYTHON_VERSION=3.8 \
    --build-arg PYTORCH_VERSION=1.9 \
    .
```

To run your docker use

```bash
docker image list
docker run --rm -it pytorch-lightning:latest bash
```

and if you do not need it anymore, just clean it:

```bash
docker image list
docker image rm pytorch-lightning:latest
```

## Run docker image with GPUs

To run docker image with access to you GPUs you need to install
```bash
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

and later run the docker image with `--gpus all` so for example

```
docker run --rm -it --gpus all pytorchlightning/pytorch_lightning:base-cuda-py3.7-torch1.6
```

## Run Jupyter server

Inspiration comes from https://u.group/thinking/how-to-put-jupyter-notebooks-in-a-dockerfile

1. Build the docker image:
    ```bash
    docker image build \
        -t pytorch-lightning:v1.3.1 \
        -f dockers/nvidia/Dockerfile \
        --build-arg LIGHTNING_VERSION=1.3.1 \
        .
    ```
2. start the server and map ports:
    ```bash
    docker run --rm -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -p 8888:8888 pytorch-lightning:v1.3.1
    ```
3. Connect in local browser:
    - copy the generated path e.g. `http://hostname:8888/?token=0719fa7e1729778b0cec363541a608d5003e26d4910983c6`
    - replace the `hostname` by `localhost`
