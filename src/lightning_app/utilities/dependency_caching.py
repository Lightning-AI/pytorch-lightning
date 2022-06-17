import hashlib
from pathlib import Path


def get_hash(path: Path, chunk_num_blocks: int = 128) -> str:
    """Get the hash of a file."""
    h = hashlib.blake2b(digest_size=20)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_num_blocks * h.block_size), b""):
            h.update(chunk)
    return h.hexdigest()
