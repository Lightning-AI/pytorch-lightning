"""Utilities for traversing the tree of components in an app."""
from typing import Type, TYPE_CHECKING

import lightning_app

if TYPE_CHECKING:
    from lightning_app.utilities.types import Component, ComponentTuple


def breadth_first(root: "Component", types: Type["ComponentTuple"] = None):
    """Returns a generator that walks through the tree of components breadth-first.

    Arguments:
        root: The root component of the tree
        types: If provided, only the component types in this list will be visited.
    """
    yield from _BreadthFirstVisitor(root, types)


def depth_first(root: "Component", types: Type["ComponentTuple"] = None):
    """Returns a generator that walks through the tree of components depth-first.

    Arguments:
        root: The root component of the tree
        types: If provided, only the component types in this list will be visited.
    """
    yield from _DepthFirstVisitor(root, types)


class _BreadthFirstVisitor:
    def __init__(self, root: "Component", types: Type["ComponentTuple"] = None) -> None:
        self.queue = [root]
        self.types = types

    def __iter__(self):
        return self

    def __next__(self):
        while self.queue:
            component = self.queue.pop(0)

            if isinstance(component, lightning_app.LightningFlow):
                self.queue += list(component.flows.values())
                self.queue += component.works(recurse=False)

            if any(isinstance(component, t) for t in self.types):
                return component

        raise StopIteration


class _DepthFirstVisitor:
    def __init__(self, root: "Component", types: Type["ComponentTuple"] = None) -> None:
        self.stack = [root]
        self.types = types

    def __iter__(self):
        return self

    def __next__(self):
        while self.stack:
            component = self.stack.pop()

            if isinstance(component, lightning_app.LightningFlow):
                self.stack += list(component.flows.values())
                self.stack += component.works(recurse=False)

            if any(isinstance(component, t) for t in self.types):
                return component

        raise StopIteration
