# PyTorch-Lightning Tests
Most PL tests train a full MNIST model under various trainer conditions (ddp, ddp2+amp, etc...).
This provides testing for most combinations of important settings.
The tests expect the model to perform to a reasonable degree of testing accuracy to pass.

## Running tests
The automatic travis tests ONLY run CPU-based tests. Although these cover most of the use cases,
run on a 2-GPU machine to validate the full test-suite.


To run all tests do the following:
```bash
git clone https://github.com/williamFalcon/pytorch-lightning
cd pytorch-lightning

# install module locally
pip install -e .

# install dev deps
pip install -r requirements.txt

# run tests
py.test -v
```

To test models that require GPU make sure to run the above command on a GPU machine.
The GPU machine must have:
1. At least 2 GPUs.
2. [NVIDIA-apex](https://github.com/NVIDIA/apex#linux) installed.


## Running Coverage   
Make sure to run coverage on a GPU machine with at least 2 GPUs and NVIDIA apex installed. 

```bash
cd pytorch-lightning

# generate coverage 
pip install coverage
coverage run --source pytorch_lightning -m py.test pytorch_lightning tests examples -v --doctest-modules

# print coverage stats
coverage report -m

# exporting resulys
coverage xml
codecov -t 17327163-8cca-4a5d-86c8-ca5f2ef700bc  -v
```


