import torch

from pathlib import Path


def load(path_or_url: str, map_location=None):
    if Path(path_or_url).is_file():
        # local file
        return torch.load(path_or_url, map_location=map_location)
    return torch.hub.load_state_dict_from_url(path_or_url, map_location=map_location)
