import datetime
import os
import re

# set paths
_PATH_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
_PATH_INFO = os.path.join(_PATH_ROOT, "pytorch_lightning", "__about__.py")

# get today date
now = datetime.datetime.now()
now_date = now.strftime("%Y%m%d")

print(f"prepare init '{_PATH_INFO}' - replace version by {now_date}")
with open(_PATH_INFO, "r") as fp:
    init = fp.read()
init = re.sub(r'__version__ = [\d\.\w\'"]+', f'__version__ = "{now_date}"', init)
with open(_PATH_INFO, "w") as fp:
    fp.write(init)
