import io
import os
import urllib
from contextlib import contextmanager
from subprocess import Popen
from typing import Any, Callable, Optional, Tuple

from lightning.data.constants import _IS_IN_STUDIO


def get_worker_rank() -> Optional[str]:
    return os.getenv("DATA_OPTIMIZER_GLOBAL_RANK")


def catch(func: Callable) -> Callable:
    def _wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, Optional[Exception]]:
        try:
            return func(*args, **kwargs), None
        except Exception as e:
            return None, e
    return _wrapper

# Credit to the https://github.com/rom1504/img2dataset Github repo
# The code was taken from there. It has a MIT License.

def make_request(
    url: str,
    timeout: int = 10,
    user_agent_token: str = "pytorch-lightning",
) -> io.BytesIO:
    """Download an image with urllib."""
    user_agent_string = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
    if user_agent_token:
        user_agent_string += f" (compatible; {user_agent_token}; +https://github.com/Lightning-AI/pytorch-lightning)"

    with urllib.request.urlopen(urllib.request.Request(url, data=None, headers={"User-Agent": user_agent_string}), timeout=timeout) as r:  # noqa: E501, S310
        img_stream = io.BytesIO(r.read())
    return img_stream


@contextmanager
def optimize_dns_context(enable: bool) -> Any:
    optimize_dns(enable)
    try:
        yield
        optimize_dns(False)  # always disable the optimize DNS
    except Exception as e:
        optimize_dns(False)  # always disable the optimize DNS
        raise e

def optimize_dns(enable: bool) -> None:
    if not _IS_IN_STUDIO:
        return

    with open("/etc/resolv.conf") as f:
        lines = f.readlines()

    if (
        (enable and any("127.0.0.53" in line for line in lines))
        or (not enable and any("127.0.0.1" in line for line in lines))
    ): # noqa E501
        Popen(f"sudo /home/zeus/miniconda3/envs/cloudspace/bin/python -c 'from lightning.data.processing.utilities import _optimize_dns; _optimize_dns({enable})'", shell=True).wait() # noqa E501

def _optimize_dns(enable: bool) -> None:
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
