from typing import Any

from torch.utils._pytree import SUPPORTED_NODES, LeafSpec, PyTree, TreeSpec, _get_node_type, tree_unflatten


def _is_leaf_or_primitive_container(pytree: PyTree) -> bool:
    """Customized :func:`torch.utils._pytree._is_leaf` to avoid flattening containers of primitives."""
    is_leaf = _get_node_type(pytree) not in SUPPORTED_NODES
    if is_leaf:
        return True

    node_type = _get_node_type(pytree)
    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    child_pytrees, _ = flatten_fn(pytree)
    return all(isinstance(child, (int, float, str)) for child in child_pytrees)


def _tree_flatten(pytree: PyTree) -> tuple[list[Any], TreeSpec]:
    """Copy of :func:`torch.utils._pytree.tree_flatten` using our custom leaf function."""
    if _is_leaf_or_primitive_container(pytree):
        return [pytree], LeafSpec()

    node_type = _get_node_type(pytree)
    flatten_fn = SUPPORTED_NODES[node_type].flatten_fn
    child_pytrees, context = flatten_fn(pytree)

    result: list[Any] = []
    children_specs: list[TreeSpec] = []
    for child in child_pytrees:
        flat, child_spec = _tree_flatten(child)
        result += flat
        children_specs.append(child_spec)

    return result, TreeSpec(node_type, context, children_specs)


def _map_and_unflatten(fn: Any, values: list[Any], spec: TreeSpec) -> PyTree:
    """Utility function to apply a function and unflatten it."""
    return tree_unflatten([fn(i) for i in values], spec)
