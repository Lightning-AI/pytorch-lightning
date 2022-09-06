# Copyright The PyTorch Lightning team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
from functools import partial
from typing import Type
from unittest.mock import Mock


def is_overridden(method_name: str, instance: object, parent: Type[object]) -> bool:
    instance_attr = getattr(instance, method_name, None)
    if instance_attr is None:
        return False
    # `functools.wraps()` support
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

    return instance_attr.__code__ != parent_attr.__code__
