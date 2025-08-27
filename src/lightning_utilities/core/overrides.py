# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
from functools import partial
from unittest.mock import Mock


def is_overridden(method_name: str, instance: object, parent: type[object]) -> bool:
    """Return True if ``instance`` overrides ``parent.method_name``.

    Supports functions wrapped with ``functools.wraps``, context managers (``__wrapped__``),
    ``unittest.mock.Mock(wraps=...)``, and ``functools.partial``. If the parent does not define
    ``method_name``, a ``ValueError`` is raised.

    Args:
        method_name: The name of the method to check.
        instance: The object instance to inspect.
        parent: The parent class that declares the original method.

    Returns:
        True if the method implementation on the instance differs from the parent's; otherwise False.

    """
    instance_attr = getattr(instance, method_name, None)
    if instance_attr is None:
        return False
    # `functools.wraps()` and `@contextmanager` support
    if hasattr(instance_attr, "__wrapped__"):
        instance_attr = instance_attr.__wrapped__
    # `Mock(wraps=...)` support
    if isinstance(instance_attr, Mock):
        # access the wrapped function
        instance_attr = instance_attr._mock_wraps
    # `partial` support
    elif isinstance(instance_attr, partial):
        instance_attr = instance_attr.func
    if instance_attr is None:
        return False

    parent_attr = getattr(parent, method_name, None)
    if parent_attr is None:
        raise ValueError("The parent should define the method")
    # `@contextmanager` support
    if hasattr(parent_attr, "__wrapped__"):
        parent_attr = parent_attr.__wrapped__

    return instance_attr.__code__ != parent_attr.__code__
