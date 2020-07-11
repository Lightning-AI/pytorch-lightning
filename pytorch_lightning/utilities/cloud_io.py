import torch

from pathlib import Path
from urllib.parse import urlparse


def load(path_or_url: str, map_location=None):
    if urlparse(path_or_url).scheme == '' or Path(path_or_url).drive:  # no scheme or with a drive letter
        return torch.load(path_or_url, map_location=map_location)
    else:
        return torch.hub.load_state_dict_from_url(path_or_url, map_location=map_location)
