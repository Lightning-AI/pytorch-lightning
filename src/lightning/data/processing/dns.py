from contextlib import contextmanager
from subprocess import Popen

from lightning.data.constants import _IS_IN_STUDIO


@contextmanager
def optimize_dns_context(enable):
    optimize_dns(enable)
    try:
        yield
        optimize_dns(False)  # always disable the optimize DNS
    except Exception as e:
        optimize_dns(False)  # always disable the optimize DNS
        raise e

def optimize_dns(enable):
    if not _IS_IN_STUDIO:
        return

    with open("/etc/resolv.conf") as f:
        lines = f.readlines()

    if (enable and any("127.0.0.53" in line for line in lines)) or (not enable and any("127.0.0.1" in line for line in lines)): # noqa E501
        Popen(f"sudo /home/zeus/miniconda3/envs/cloudspace/bin/python -c 'from lightning.data.processing import _optimize_dns; _optimize_dns({enable})'", shell=True).wait() # noqa E501

def _optimize_dns(enable=False):
    with open("/etc/resolv.conf") as f:
        lines = f.readlines()

    write_lines = []
    for line in lines:
        if "nameserver 127" in line:
            if enable:
                write_lines.append('nameserver 127.0.0.1\n')
            else:
                write_lines.append('nameserver 127.0.0.53\n')
        else:
            write_lines.append(line)

    with open("/etc/resolv.conf", "w") as f:
        for line in write_lines:
            f.write(line)
