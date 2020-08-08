## Builds

You can build it on your own, note it takes lots of time, be prepared.

```bash
git clone <git-repository>
docker image build -t pytorch-lightning:latest -f dockers/conda/Dockerfile .
```

or with specific arguments

```bash
git clone <git-repository>
docker image build \
    -t pytorch-lightning:py3.8 \
    -f dockers/conda/Dockerfile \
    --build-arg PYTHON_VERSION=3.8 \
    --build-arg PYTORCH_VERSION=1.4 \
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