import subprocess
from typing import Tuple


def _get_available_amount_of_bytes() -> Tuple[int, int]:
    """Returns the available amount of bytes on the machine."""
    result = subprocess.check_output(["df", "-k", "/"])
    result = [v for v in result.decode("utf-8").split("\n")[1].split(" ") if v != ""]
    available = int(result[3])
    return available
