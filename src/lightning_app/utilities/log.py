from pathlib import Path

from lightning_app.storage.path import storage_root_dir


def get_frontend_logfile(filename: str = "logs.log") -> Path:
    log_dir = Path(storage_root_dir(), "frontend")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / filename
    return log_file
