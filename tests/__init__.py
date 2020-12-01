import os

import numpy as np

TEST_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(TEST_ROOT)
TEMP_PATH = os.path.join(PROJECT_ROOT, 'test_temp')

OS_PYTHONPATH = os.getenv('PYTHONPATH', default='')
if PROJECT_ROOT not in OS_PYTHONPATH:
    OS_PYTHONPATH += ';' + PROJECT_ROOT if OS_PYTHONPATH else PROJECT_ROOT
    os.putenv('PYTHONPATH', OS_PYTHONPATH)

# generate a list of random seeds for each test
RANDOM_PORTS = list(np.random.randint(12000, 19000, 1000))

if not os.path.isdir(TEMP_PATH):
    os.mkdir(TEMP_PATH)
