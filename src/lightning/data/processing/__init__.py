from typing import Callable, Any

def catch(func: Callable) -> Callable:
    def _wrapper(*args: Any, **kwargs: Any):
        try:
            return func(*args, **kwargs), None
        except Exception as e:
            return None, e
    return _wrapper