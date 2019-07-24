# Pytorch-Lightning Tests

## Running tests

To run all tests do the following:
```bash
git clone https://github.com/williamFalcon/pytorch-lightning
cd pytorch-lightning

# install module locally
pip install -e .

# install dev deps
pip install -r requirements.txt

# run tests
py.test

# or to generate coverage 
pip install coverage
coverage run tests/test_models.py   
```

To test models that require GPU make sure to run the above command on a GPU machine.
The GPU machine must have:
1. At least 2 GPUs.
2. [NVIDIA-apex](https://github.com/NVIDIA/apex#linux) installed.


### test_models.py
This file fits a tiny model on MNIST using these different set-ups.
1. CPU only.
2. Single GPU with DP.
3. Multiple (2) GPUs using DP.
3. Multiple (2) GPUs using DDP.
3. Multiple (2) GPUs using DP + apex (for 16-bit precision).
3. Multiple (2) GPUs using DDP + apex (for 16-bit precision).   


