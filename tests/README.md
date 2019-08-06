# PyTorch-Lightning Tests

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
py.test
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

For each set up it also tests:
1. model saving.
2. model loading.
3. predicting with a loaded model.
4. simulated save from HPC signal.
5. simulated load from HPC signal.

## Running Coverage   
Make sure to run coverage on a GPU machine with at least 2 GPUs and NVIDIA apex installed. 

```bash
cd pytorch-lightning

# generate coverage 
pip install coverage
coverage run tests/test_models.py   

# print coverage stats
coverage report -m   
```


