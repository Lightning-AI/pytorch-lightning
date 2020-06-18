import torch

from pathlib import Path
from urllib.parse import urlparse


def load(path_or_url: str, map_location=None):
    parsed = urlparse(path_or_url)
    if parsed.scheme == '' or Path(path_or_url).is_file():
        # no scheme or local file
        return torch.load(path_or_url, map_location=map_location)
    return torch.hub.load_state_dict_from_url(path_or_url, map_location=map_location)
