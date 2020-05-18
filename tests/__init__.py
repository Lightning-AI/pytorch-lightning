import os

import numpy as np
import torch

TEST_ROOT = os.path.dirname(__file__)
PACKAGE_ROOT = os.path.dirname(TEST_ROOT)
TEMP_PATH = os.path.join(PACKAGE_ROOT, 'test_temp')

# generate a list of random seeds for each test
RANDOM_PORTS = list(np.random.randint(12000, 19000, 1000))
ROOT_SEED = 1234
torch.manual_seed(ROOT_SEED)
np.random.seed(ROOT_SEED)
RANDOM_SEEDS = list(np.random.randint(0, 10000, 1000))

if not os.path.isdir(TEMP_PATH):
    os.mkdir(TEMP_PATH)
