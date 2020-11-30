# PyTorch-Lightning Tests
Most PL tests train a full MNIST model under various trainer conditions (ddp, ddp2+amp, etc...).
This provides testing for most combinations of important settings.
The tests expect the model to perform to a reasonable degree of testing accuracy to pass.

## Running tests
The automatic travis tests ONLY run CPU-based tests. Although these cover most of the use cases,
run on a 2-GPU machine to validate the full test-suite.


To run all tests do the following:

Install [Open MPI](https://www.open-mpi.org/) or another MPI implementation. Learn how to install Open MPI [on this page](https://www.open-mpi.org/faq/?category=building#easy-build>).

```bash
git clone https://github.com/PyTorchLightning/pytorch-lightning
cd pytorch-lightning

# install AMP support
bash requirements/install_AMP.sh

# install dev deps
pip install -r requirements/devel.txt

# run tests
py.test -v
```

To test models that require GPU make sure to run the above command on a GPU machine.
The GPU machine must have:
1. At least 2 GPUs.
2. [NVIDIA-apex](https://github.com/NVIDIA/apex#linux) installed.
3. [Horovod with NCCL](https://horovod.readthedocs.io/en/stable/gpus_include.html) support: `HOROVOD_GPU_OPERATIONS=NCCL pip install horovod`


## Running Coverage   
Make sure to run coverage on a GPU machine with at least 2 GPUs and NVIDIA apex installed. 

```bash
cd pytorch-lightning

# generate coverage (coverage is also installed as part of dev dependencies under requirements/devel.txt)
coverage run --source pytorch_lightning -m py.test pytorch_lightning tests examples -v

# print coverage stats
coverage report -m

# exporting results
coverage xml
```

## Building test image

You can build it on your own, note it takes lots of time, be prepared.
```bash
git clone <git-repository>
docker image build -t pytorch_lightning:devel-torch1.4 -f dockers/cuda-extras/Dockerfile --build-arg TORCH_VERSION=1.4 .
```
To build other versions, select different Dockerfile.
```bash
docker image list
docker run --rm -it pytorch_lightning:devel-torch1.4 bash
docker image rm pytorch_lightning:devel-torch1.4
```
