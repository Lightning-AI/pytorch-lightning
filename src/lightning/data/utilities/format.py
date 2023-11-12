_FORMAT_TO_RATIO = {
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
}


def _convert_bytes_to_int(bytes_str: str) -> int:
    """Convert human-readable byte format to an integer."""
    for suffix in _FORMAT_TO_RATIO:
        bytes_str = bytes_str.lower().strip()
        if bytes_str.lower().endswith(suffix):
            try:
                return int(float(bytes_str[0 : -len(suffix)]) * _FORMAT_TO_RATIO[suffix])
            except ValueError:
                raise ValueError(
                    f"Unsupported value/suffix {bytes_str}. Supported suffix are "
                    f'{["b"] + list(_FORMAT_TO_RATIO.keys())}.'
                )
    raise ValueError(f"The supported units are {_FORMAT_TO_RATIO.keys()}")


def _human_readable_bytes(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1000.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1000.0
    return f"{num_bytes:.1f} PB"
