from typing import Tuple
from urllib.parse import urlparse
import os.path as osp
import os
import torch
from torch.hub import _get_torch_home

import logging

logger = logging.getLogger(__name__)

torch_cache_home = _get_torch_home()
default_cache_path = osp.join(torch_cache_home, "pl_checkpoints")


def try_import_boto3():
    try:
        import boto3
    except ImportError:
        raise ImportError(f'Could not import `boto3`. Please `pip install boto3` and try again.')


def load(path_or_url: str, map_location=None):
    parsed = urlparse(path_or_url)
    if parsed.scheme == '' or Path(path_or_url).is_file():
        # no scheme or local file
        return torch.load(path_or_url, map_location=map_location)
    elif parsed.scheme == 's3':
        # AWS S3 file
        filepath = download_checkpoint_from_s3(path_or_url)
        return torch.load(filepath, map_location=map_location)
    # URL
    return torch.hub.load_state_dict_from_url(path_or_url, map_location=map_location)


def is_s3_path(path: str):
    """Checks if path is a valid S3 path"""
    return path.startswith("s3://")


def parse_s3_path(s3_path: str) -> Tuple[str, str]:
    """
    Returns bucket and key from an S3 path.
    Example: "s3://my-bucket/folder/checkpoint.ckpt" -> ("my-bucket", "folder/checkpoint.ckpt")
    """
    s3_path = urlparse(s3_path, allow_fragments=True)
    assert s3_path.scheme == 's3', f'{s3_path} is not a valid AWS S3 path. Needs to start with `s3://`'
    bucket, key = s3_path.netloc, s3_path.path
    if key.startswith('/'):
        key = key[1:]
    return bucket, key


def save_checkpoint_to_s3(bucket_name, key):
    """
    Saves a single checkpoint to an S3 path.
    Args:
        bucket_name: The name of the bucket we want to save to
        key: The rest of the s3 path.
    Returns:
    None
    """
    try_import_boto3()
    bucket = boto3.resource("s3").Bucket(bucket_name)
    bucket.upload_file(Filename=key, Key=key)


def download_checkpoint_from_s3(path_or_url: str, overwrite=False) -> str:
    """
    Downloads file from S3 and saves it in default cache path under original S3 key.
    Returns filepath where object has been downloaded.
    """
    try_import_boto3()

    # Eg "s3://bucket-name/folder/checkpoint.ckpt" --> ("bucket-name", "folder/checkpoint.ckpt")
    bucket_name, key = parse_s3_path(path_or_url)

    # ("folder", "checkpoint.ckpt")
    directory, filename = osp.split(key)

    # Make directory: '/Users/johnDoe/.cache/torch/pl_checkpoints/folder'
    directory_to_make = osp.join(default_cache_path, directory)
    os.makedirs(directory_to_make, exist_ok=True)

    # File we will download to: '/Users/johnDoe/.cache/torch/pl_checkpoints/folder/checkpoint.ckpt'
    filepath = osp.join(directory_to_make, filename)

    def _download():
        s3 = boto3.resource("s3")
        bucket = s3.Bucket(bucket_name)
        bucket.download_file(Key=key, Filename=filepath)

    if not osp.exists(filepath):
        _download()
    else:
        if overwrite:
            _download()
    return filepath


def remove_checkpoint_from_s3(bucket, key):
    """Simple remove object from S3"""
    try_import_boto3()
    s3 = boto3.resource("s3")
    obj = s3.Object(bucket, key)
    obj.delete()
