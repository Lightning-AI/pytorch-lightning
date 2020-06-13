import torch

from urllib.parse import urlparse


def load(path_or_url: str, map_location=None):
    parsed = urlparse(path_or_url)
    if parsed.scheme == "":
        # local file
        return torch.load(path_or_url, map_location=map_location)
    elif parsed.scheme == "s3":
        return load_s3_checkpoint(path_or_url, map_location=map_location)
    return torch.hub.load_state_dict_from_url(path_or_url, map_location=map_location)


def load_s3_checkpoint(checkpoint_path, map_location, **pickle_load_args):
    from torch.serialization import _legacy_load
    import pickle

    # Attempt s3fs import
    try:
        import s3fs
    except ImportError:
        raise ImportError(
            f"Tried to import `s3fs` for AWS S3 i/o, but `s3fs` is not installed. "
            f"Please `pip install s3fs` and try again."
        )

    if "encoding" not in pickle_load_args.keys():
        pickle_load_args["encoding"] = "utf-8"
    fs = s3fs.S3FileSystem()
    with fs.open(checkpoint_path, "rb") as f:
        checkpoint = _legacy_load(f, map_location, pickle, **pickle_load_args)
    return checkpoint


def save_s3_checkpoint(checkpoint, checkpoint_path):
    from torch.serialization import _legacy_save
    import pickle

    # Attempt s3fs import
    try:
        import s3fs
    except ImportError:
        raise ImportError(
            f"Tried to import `s3fs` for AWS S3 i/o, but `s3fs` is not installed. "
            f"Please `pip install s3fs` and try again."
        )

    DEFAULT_PROTOCOL = 2  # from torch.serialization.py line 19
    fs = s3fs.S3FileSystem()
    with fs.open(checkpoint_path, "wb") as f:
        checkpoint = _legacy_save(checkpoint, checkpoint_path, pickle, DEFAULT_PROTOCOL)
    return checkpoint
