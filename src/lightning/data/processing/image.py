import io
from typing import Optional, Tuple

from lightning_utilities.core.imports import RequirementCache

_HTTPX_AVAILABLE = RequirementCache("httpx")

# Credit to the https://github.com/rom1504/img2dataset Github repo
# The code was taken from there. It has a MIT License.

def _download_image(
    url: str,
    timeout: int = 10,
    user_agent_token: str = "pytorch-lightning",
) -> Tuple[Optional[io.BytesIO], Optional[Exception]]:
    """Download an image with urllib."""
    url
    img_stream = None
    user_agent_string = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
    if user_agent_token:
        user_agent_string += f" (compatible; {user_agent_token}; +https://github.com/Lightning-AI/pytorch-lightning)"
    import httpx

    try:
        with httpx.Client(http2=True) as client:
            r = client.get(url, headers={"User-Agent": user_agent_string}, timeout=timeout)
            img_stream = io.BytesIO(r.read())
        return img_stream, None
    except Exception as err:  # pylint: disable=broad-except
        if img_stream is not None:
            img_stream.close()
        return None, err


def download_image(
    url: str,
    retries: int = 0,
    timeout: int = 10,
    user_agent_token: str = "pytorch-lightning",
) -> Tuple[Optional[io.BytesIO], Optional[Exception]]:
    if not _HTTPX_AVAILABLE:
        raise ModuleNotFoundError("Please, run: `pip install httpx`.")
    for _ in range(retries + 1):
        img_stream, err = _download_image(url, timeout, user_agent_token)
        if img_stream is not None:
            return img_stream, err
    return None, err
