import os
import re

PATH_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

PATH_SETUP = os.path.join(PATH_ROOT, 'setup.py')
print(f"rename package '{PATH_SETUP}'")
with open(PATH_SETUP, 'r') as fp:
    setup = fp.read()
setup = re.sub(r'name=[\'"]pytorch-lightning[\'"]', 'name="pytorch-lightning-nightly"', setup)
with open(PATH_SETUP, 'w') as fp:
    fp.write(setup)
