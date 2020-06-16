import torch

from urllib.parse import urlparse


def load(path_or_url: str, map_location=None):
    parsed = urlparse(path_or_url)
    if parsed.scheme == '':
        # local file
        return torch.load(path_or_url, map_location=map_location)
    return torch.hub.load_state_dict_from_url(path_or_url, map_location=map_location)
