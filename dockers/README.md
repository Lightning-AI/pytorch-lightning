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
    -t pytorch-lightning:py3.8-pt1.6 \
    -f dockers/base-cuda/Dockerfile \
    --build-arg PYTHON_VERSION=3.8 \
    --build-arg PYTORCH_VERSION=1.6 \
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

### Run docker image with GPUs

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
