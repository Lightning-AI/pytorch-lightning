from pathlib import Path

from lightning_app.storage.path import _storage_root_dir


def get_logfile(filename: str = "logs.log") -> Path:
    log_dir = Path(_storage_root_dir(), "frontend")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / filename
    return log_file
