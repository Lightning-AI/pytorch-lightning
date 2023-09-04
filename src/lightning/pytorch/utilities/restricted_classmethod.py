from typing import Callable, Concatenate, Generic, ParamSpec, TypeVar

_T = TypeVar("_T")  # type of the method owner
_P = ParamSpec("_P")  # parameters of the decorated method
_R_co = TypeVar("_R_co", covariant=True)  # return type of the decorated method


class restricted_classmethod(Generic[_T, _P, _R_co]):
    """
    Custom `classmethod` that raises an exception when the classmethod is
    called on an instance and not the class type.
    """

    def __init__(self, method: Callable[Concatenate[_T, _P], _R_co]) -> None:
        self.method = method

    def __get__(self, instance: _T, cls: type[_T]) -> Callable[_P, _R_co]:
        if instance is not None:
            raise TypeError(
                f"The classmethod `{cls.__name__}.{self.method.__name__}` cannot be called on an instance. Please "
                f"call it on the class type and make sure the return value is used."
            )
        return self.method.__get__(cls, cls)
