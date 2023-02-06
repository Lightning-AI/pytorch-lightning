# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for traversing the tree of components in an app."""
from typing import Type, TYPE_CHECKING

import lightning.app

if TYPE_CHECKING:
    from lightning.app.utilities.types import Component, ComponentTuple


def breadth_first(root: "Component", types: Type["ComponentTuple"] = None):
    """Returns a generator that walks through the tree of components breadth-first.

    Arguments:
        root: The root component of the tree
        types: If provided, only the component types in this list will be visited.
    """
    yield from _BreadthFirstVisitor(root, types)


class _BreadthFirstVisitor:
    def __init__(self, root: "Component", types: Type["ComponentTuple"] = None) -> None:
        self.queue = [root]
        self.types = types

    def __iter__(self):
        return self

    def __next__(self):
        from lightning.app.structures import Dict

        while self.queue:
            component = self.queue.pop(0)

            if isinstance(component, lightning.app.LightningFlow):
                components = [getattr(component, el) for el in sorted(component._flows)]
                for struct_name in sorted(component._structures):
                    structure = getattr(component, struct_name)
                    if isinstance(structure, Dict):
                        values = sorted(structure.items(), key=lambda x: x[0])
                    else:
                        values = sorted(((v.name, v) for v in structure), key=lambda x: x[0])
                    for _, value in values:
                        if isinstance(value, lightning.app.LightningFlow):
                            components.append(value)
                self.queue += components
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

            if isinstance(component, lightning.app.LightningFlow):
                self.stack += list(component.flows.values())
                self.stack += component.works(recurse=False)

            if any(isinstance(component, t) for t in self.types):
                return component

        raise StopIteration
