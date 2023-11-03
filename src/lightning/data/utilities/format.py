def _human_readable_bytes(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"):
        if abs(num_bytes) < 1000.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1000.0
    return f"{num_bytes:.1f} YB"
