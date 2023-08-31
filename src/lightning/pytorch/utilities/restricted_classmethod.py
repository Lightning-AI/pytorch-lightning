from typing import Callable, Concatenate, Generic, ParamSpec, TypeVar


class RestrictedClassmethodError(Exception):
    """
    This exception raised when a `restricted_classmethod` is invoked on an instance
    instead of a class type.
    """


_T = TypeVar("_T")  # type of the method owner
_P = ParamSpec("_P")  # parameters of the decorated method
_R_co = TypeVar("_R_co", covariant=True)  # return type of the decorated method


class restricted_classmethod(Generic[_T, _P, _R_co]):
    """
    Custom `classmethod` that emits a warning when the classmethod is
    called on an instance and not the class type.
    """

    def __init__(self, method: Callable[Concatenate[_T, _P], _R_co]) -> None:
        self.method = method

    def __get__(self, instance: _T, cls: type[_T]) -> Callable[_P, _R_co]:
        if instance is not None:
            raise RestrictedClassmethodError(
                f"The classmethod `{cls.__name__}.{self.method.__name__}` cannot be invoked on an instance. Please "
                f"invoke it on the class type and make sure the return value is used."
            )
        return self.method.__get__(cls, cls)
