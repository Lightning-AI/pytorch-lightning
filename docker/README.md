## Builds

You can build it on your own, note it takes lots of time, be prepared.
```bash
git clone <git-repository>
docker image build -t pytorch-lightning:py36 -f docker/Dockerfile --build-arg PYTHON_VERSION=3.6 .
```
To build other versions, select different Dockerfile.
```bash
docker image list
docker run --rm -it pytorch-lightning:py36 bash
docker image rm pytorch-lightning:py36
```