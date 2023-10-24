from typing import Any, Callable, Tuple, Type, TypeVar, Union

from torch.utils._pytree import PyTree

from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_1_13

if _TORCH_GREATER_EQUAL_1_13:
    from torch.utils._pytree import map_only, tree_map_only
else:
    from torch.utils._pytree import tree_map

    T = TypeVar("T")
    R = TypeVar("R")
    TypeAny = Union[Type[Any], Tuple[Type[Any], ...]]
    MapOnlyFn = Callable[[T], Callable[[Any], Any]]
    FnAny = Callable[[Any], R]

    def map_only(ty: TypeAny) -> MapOnlyFn[FnAny[Any]]:
        def deco(f: Callable[[T], Any]) -> Callable[[Any], Any]:
            def inner(x: T) -> Any:
                if isinstance(x, ty):
                    return f(x)
                return x

            return inner

        return deco

    def tree_map_only(ty: TypeAny, fn: FnAny[Any], pytree: PyTree) -> PyTree:
        return tree_map(map_only(ty)(fn), pytree)
