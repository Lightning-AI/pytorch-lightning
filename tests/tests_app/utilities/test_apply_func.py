# Copyright The PyTorch Lightning team.
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
import dataclasses
import numbers
from collections import defaultdict, namedtuple, OrderedDict
from dataclasses import InitVar
from typing import Any, ClassVar, List, Optional

import pytest

from lightning_app.utilities.apply_func import apply_to_collection
from lightning_app.utilities.exceptions import MisconfigurationException
from lightning_app.utilities.imports import _is_numpy_available, _is_torch_available

if _is_torch_available():
    import torch

if _is_numpy_available():
    import numpy as np


@pytest.mark.skipif(not (_is_torch_available() and _is_numpy_available()), reason="Requires torch and numpy")
def test_recursive_application_to_collection():
    ntc = namedtuple("Foo", ["bar"])

    @dataclasses.dataclass
    class Feature:
        input_ids: torch.Tensor
        segment_ids: np.ndarray

        def __eq__(self, o: object) -> bool:
            if not isinstance(o, Feature):
                return NotImplemented
            else:
                return torch.equal(self.input_ids, o.input_ids) and np.equal(self.segment_ids, o.segment_ids).all()

    @dataclasses.dataclass
    class ModelExample:
        example_ids: List[str]
        feature: Feature
        label: torch.Tensor
        some_constant: int = dataclasses.field(init=False)

        def __post_init__(self):
            self.some_constant = 7

        def __eq__(self, o: object) -> bool:
            if not isinstance(o, ModelExample):
                return NotImplemented
            else:
                return (
                    self.example_ids == o.example_ids
                    and self.feature == o.feature
                    and torch.equal(self.label, o.label)
                    and self.some_constant == o.some_constant
                )

    @dataclasses.dataclass
    class WithClassVar:
        class_var: ClassVar[int] = 0
        dummy: Any

        def __eq__(self, o: object) -> bool:
            if not isinstance(o, WithClassVar):
                return NotImplemented
            elif isinstance(self.dummy, torch.Tensor):
                return torch.equal(self.dummy, o.dummy)
            else:
                return self.dummy == o.dummy

    @dataclasses.dataclass
    class WithInitVar:
        dummy: Any
        override: InitVar[Optional[Any]] = None

        def __post_init__(self, override: Optional[Any]):
            if override is not None:
                self.dummy = override

        def __eq__(self, o: object) -> bool:
            if not isinstance(o, WithInitVar):
                return NotImplemented
            elif isinstance(self.dummy, torch.Tensor):
                return torch.equal(self.dummy, o.dummy)
            else:
                return self.dummy == o.dummy

    @dataclasses.dataclass
    class WithClassAndInitVar:
        class_var: ClassVar[torch.Tensor] = torch.tensor(0)
        dummy: Any
        override: InitVar[Optional[Any]] = torch.tensor(1)

        def __post_init__(self, override: Optional[Any]):
            if override is not None:
                self.dummy = override

        def __eq__(self, o: object) -> bool:
            if not isinstance(o, WithClassAndInitVar):
                return NotImplemented
            elif isinstance(self.dummy, torch.Tensor):
                return torch.equal(self.dummy, o.dummy)
            else:
                return self.dummy == o.dummy

    model_example = ModelExample(
        example_ids=["i-1", "i-2", "i-3"],
        feature=Feature(input_ids=torch.tensor([1.0, 2.0, 3.0]), segment_ids=np.array([4.0, 5.0, 6.0])),
        label=torch.tensor([7.0, 8.0, 9.0]),
    )

    to_reduce = {
        "a": torch.tensor([1.0]),  # Tensor
        "b": [torch.tensor([2.0])],  # list
        "c": (torch.tensor([100.0]),),  # tuple
        "d": ntc(bar=5.0),  # named tuple
        "e": np.array([10.0]),  # numpy array
        "f": "this_is_a_dummy_str",  # string
        "g": 12.0,  # number
        "h": Feature(input_ids=torch.tensor([1.0, 2.0, 3.0]), segment_ids=np.array([4.0, 5.0, 6.0])),  # dataclass
        "i": model_example,  # nested dataclass
        "j": WithClassVar(torch.arange(3)),  # dataclass with class variable
        "k": WithInitVar("this_gets_overridden", torch.tensor([2.0])),  # dataclass with init-only variable
        "l": WithClassAndInitVar(model_example, None),  # nested dataclass with class and init-only variables
    }

    model_example_result = ModelExample(
        example_ids=["i-1", "i-2", "i-3"],
        feature=Feature(input_ids=torch.tensor([2.0, 4.0, 6.0]), segment_ids=np.array([8.0, 10.0, 12.0])),
        label=torch.tensor([14.0, 16.0, 18.0]),
    )

    expected_result = {
        "a": torch.tensor([2.0]),
        "b": [torch.tensor([4.0])],
        "c": (torch.tensor([200.0]),),
        "d": ntc(bar=torch.tensor([10.0])),
        "e": np.array([20.0]),
        "f": "this_is_a_dummy_str",
        "g": 24.0,
        "h": Feature(input_ids=torch.tensor([2.0, 4.0, 6.0]), segment_ids=np.array([8.0, 10.0, 12.0])),
        "i": model_example_result,
        "j": WithClassVar(torch.arange(0, 6, 2)),
        "k": WithInitVar(torch.tensor([4.0])),
        "l": WithClassAndInitVar(model_example_result, None),
    }

    reduced = apply_to_collection(to_reduce, (torch.Tensor, numbers.Number, np.ndarray), lambda x: x * 2)

    assert isinstance(reduced, dict), "Type Consistency of dict not preserved"
    assert all(x in reduced for x in to_reduce), "Not all entries of the dict were preserved"
    assert all(
        isinstance(reduced[k], type(expected_result[k])) for k in to_reduce
    ), "At least one type was not correctly preserved"

    assert isinstance(reduced["a"], torch.Tensor), "Reduction Result of a Tensor should be a Tensor"
    assert torch.equal(expected_result["a"], reduced["a"]), "Reduction of a tensor does not yield the expected value"

    assert isinstance(reduced["b"], list), "Reduction Result of a list should be a list"
    assert all(
        torch.equal(x, y) for x, y in zip(reduced["b"], expected_result["b"])
    ), "At least one value of list reduction did not come out as expected"

    assert isinstance(reduced["c"], tuple), "Reduction Result of a tuple should be a tuple"
    assert all(
        torch.equal(x, y) for x, y in zip(reduced["c"], expected_result["c"])
    ), "At least one value of tuple reduction did not come out as expected"

    assert isinstance(reduced["d"], ntc), "Type Consistency for named tuple not given"
    assert isinstance(
        reduced["d"].bar, numbers.Number
    ), "Failure in type promotion while reducing fields of named tuples"
    assert reduced["d"].bar == expected_result["d"].bar

    assert isinstance(reduced["e"], np.ndarray), "Type Promotion in reduction of numpy arrays failed"
    assert reduced["e"] == expected_result["e"], "Reduction of numpy array did not yield the expected result"

    assert isinstance(reduced["f"], str), "A string should not be reduced"
    assert reduced["f"] == expected_result["f"], "String not preserved during reduction"

    assert isinstance(reduced["g"], numbers.Number), "Reduction of a number should result in a number"
    assert reduced["g"] == expected_result["g"], "Reduction of a number did not yield the desired result"

    def _assert_dataclass_reduction(actual, expected, dataclass_type: str = ""):
        assert dataclasses.is_dataclass(actual) and not isinstance(
            actual, type
        ), f"Reduction of a {dataclass_type} dataclass should result in a dataclass"
        for field in dataclasses.fields(actual):
            if dataclasses.is_dataclass(field.type):
                _assert_dataclass_reduction(getattr(actual, field.name), getattr(expected, field.name), "nested")
        assert actual == expected, f"Reduction of a {dataclass_type} dataclass did not yield the desired result"

    _assert_dataclass_reduction(reduced["h"], expected_result["h"])

    _assert_dataclass_reduction(reduced["i"], expected_result["i"])

    dataclass_type = "ClassVar-containing"
    _assert_dataclass_reduction(reduced["j"], expected_result["j"], dataclass_type)
    assert WithClassVar.class_var == 0, f"Reduction of a {dataclass_type} dataclass should not change the class var"

    _assert_dataclass_reduction(reduced["k"], expected_result["k"], "InitVar-containing")

    dataclass_type = "Class-and-InitVar-containing"
    _assert_dataclass_reduction(reduced["l"], expected_result["l"], dataclass_type)
    assert torch.equal(
        WithClassAndInitVar.class_var, torch.tensor(0)
    ), f"Reduction of a {dataclass_type} dataclass should not change the class var"

    # mapping support
    reduced = apply_to_collection({"a": 1, "b": 2}, int, lambda x: str(x))
    assert reduced == {"a": "1", "b": "2"}
    reduced = apply_to_collection(OrderedDict([("b", 2), ("a", 1)]), int, lambda x: str(x))
    assert reduced == OrderedDict([("b", "2"), ("a", "1")])

    # custom mappings
    class _CustomCollection(dict):
        def __init__(self, initial_dict):
            super().__init__(initial_dict)

    to_reduce = _CustomCollection({"a": 1, "b": 2, "c": 3})
    reduced = apply_to_collection(to_reduce, int, lambda x: str(x))
    assert reduced == _CustomCollection({"a": "1", "b": "2", "c": "3"})

    # defaultdict
    to_reduce = defaultdict(int, {"a": 1, "b": 2, "c": 3})
    reduced = apply_to_collection(to_reduce, int, lambda x: str(x))
    assert reduced == defaultdict(int, {"a": "1", "b": "2", "c": "3"})


def test_apply_to_collection_include_none():
    to_reduce = [1, 2, 3.4, 5.6, 7, (8, 9.1, {10: 10})]

    def fn(x):
        if isinstance(x, float):
            return x

    reduced = apply_to_collection(to_reduce, (int, float), fn)
    assert reduced == [None, None, 3.4, 5.6, None, (None, 9.1, {10: None})]

    reduced = apply_to_collection(to_reduce, (int, float), fn, include_none=False)
    assert reduced == [3.4, 5.6, (9.1, {})]


@pytest.mark.skipif(not _is_torch_available(), reason="Requires torch and numpy")
def test_apply_to_collection_frozen_dataclass():
    @dataclasses.dataclass(frozen=True)
    class Foo:
        input: torch.Tensor

    foo = Foo(torch.tensor(0))

    with pytest.raises(MisconfigurationException, match="frozen dataclass was passed"):
        apply_to_collection(foo, torch.Tensor, lambda t: t.to(torch.int))
