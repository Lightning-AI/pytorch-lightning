import datetime
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

# get today date
now = datetime.datetime.now()
now_date = now.strftime("%Y%m%d")
PATH_INIT = os.path.join(PATH_ROOT, 'pytorch_lightning', '__init__.py')
print(f"prepare init '{PATH_INIT}' - replace version by {now_date}")
with open(PATH_INIT, 'r') as fp:
    init = fp.read()
init = re.sub(r'__version__ = [\d\.rc\'"]+', f'__version__ = "{now_date}"', init)
with open(PATH_INIT, 'w') as fp:
    fp.write(init)
