"""
Implements a decorator @_restricted_classmethod that is a drop-in replacement for
@classmethod, but raises an exception when the decorated method is called on an instance
instead of a class type.
"""
import inspect
import os
from types import MethodType
from typing import Callable, Concatenate, Generic, ParamSpec, TYPE_CHECKING, TypeVar

_T = TypeVar("_T")  # type of the method owner
_P = ParamSpec("_P")  # parameters of the decorated method
_R_co = TypeVar("_R_co", covariant=True)  # return type of the decorated method


class _restricted_classmethod_impl(Generic[_T, _P, _R_co]):
    """
    Custom `classmethod` that raises an exception when the classmethod is
    called on an instance and not the class type.
    """

    def __init__(self, method: Callable[Concatenate[_T, _P], _R_co]) -> None:
        self.method = method

    def __get__(self, instance: _T | None, cls: type[_T]) -> Callable[_P, _R_co]:
        # Workaround for https://github.com/pytorch/pytorch/issues/67146
        is_scripting = any(os.path.join("torch", "jit") in frameinfo.filename for frameinfo in inspect.stack())
        if instance is not None and not is_scripting:
            raise TypeError(
                f"The classmethod `{cls.__name__}.{self.method.__name__}` cannot be called on an instance. "
                f"Please call it on the class type and make sure the return value is used."
            )
        return MethodType(self.method, cls)


# trick static type checkers into thinking it's a @classmethod
# https://github.com/microsoft/pyright/issues/5865
_restricted_classmethod = classmethod if TYPE_CHECKING else _restricted_classmethod_impl
