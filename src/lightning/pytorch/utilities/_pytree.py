from typing import Any, List, Tuple

from torch.utils._pytree import _get_node_type, LeafSpec, PyTree, SUPPORTED_NODES, tree_unflatten, TreeSpec


def _is_leaf(pytree: PyTree) -> bool:
    is_leaf = _get_node_type(pytree) not in SUPPORTED_NODES.keys()
    if is_leaf:
        return True
    # MODIFICATION: avoid flattening lists of primitives
    node_type = _get_node_type(pytree)
    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    child_pytrees, context = flatten_fn(pytree)
    # TODO: extra types?
    return all(isinstance(child, (int, float, str)) for child in child_pytrees)


def _tree_flatten(pytree: PyTree) -> Tuple[List[Any], TreeSpec]:
    # COPIED TO USE OUR MODIFIED `_is_leaf`
    if _is_leaf(pytree):
        return [pytree], LeafSpec()

    node_type = _get_node_type(pytree)
    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    child_pytrees, context = flatten_fn(pytree)

    result: List[Any] = []
    children_specs: List["TreeSpec"] = []
    for child in child_pytrees:
        flat, child_spec = _tree_flatten(child)
        result += flat
        children_specs.append(child_spec)

    return result, TreeSpec(node_type, context, children_specs)


def _map_and_unflatten(fn: Any, values: List[Any], spec: TreeSpec) -> PyTree:
    return tree_unflatten([fn(i) for i in values], spec)
