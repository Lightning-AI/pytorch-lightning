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
from collections import namedtuple, OrderedDict
from typing import List

import numpy as np
import pytest
import torch

from pytorch_lightning.utilities.apply_func import apply_to_collection, apply_to_collections, move_data_to_device


def test_recursive_application_to_collection():
    ntc = namedtuple("Foo", ["bar"])

    @dataclasses.dataclass
    class Feature:
        input_ids: torch.Tensor
        segment_ids: np.ndarray

    @dataclasses.dataclass
    class ModelExample:
        example_ids: List[str]
        feature: Feature
        label: torch.Tensor

    to_reduce = {
        "a": torch.tensor([1.0]),  # Tensor
        "b": [torch.tensor([2.0])],  # list
        "c": (torch.tensor([100.0]),),  # tuple
        "d": ntc(bar=5.0),  # named tuple
        "e": np.array([10.0]),  # numpy array
        "f": "this_is_a_dummy_str",  # string
        "g": 12.0,  # number
        "h": Feature(input_ids=torch.tensor([1.0, 2.0, 3.0]), segment_ids=np.array([4.0, 5.0, 6.0])),  # dataclass
        "i": ModelExample(
            example_ids=["i-1", "i-2", "i-3"],
            feature=Feature(input_ids=torch.tensor([1.0, 2.0, 3.0]), segment_ids=np.array([4.0, 5.0, 6.0])),
            label=torch.tensor([7.0, 8.0, 9.0]),
        ),  # nested dataclass
    }

    expected_result = {
        "a": torch.tensor([2.0]),
        "b": [torch.tensor([4.0])],
        "c": (torch.tensor([200.0]),),
        "d": ntc(bar=torch.tensor([10.0])),
        "e": np.array([20.0]),
        "f": "this_is_a_dummy_str",
        "g": 24.0,
        "h": Feature(input_ids=torch.tensor([2.0, 4.0, 6.0]), segment_ids=np.array([8.0, 10.0, 12.0])),
        "i": ModelExample(
            example_ids=["i-1", "i-2", "i-3"],
            feature=Feature(input_ids=torch.tensor([2.0, 4.0, 6.0]), segment_ids=np.array([8.0, 10.0, 12.0])),
            label=torch.tensor([14.0, 16.0, 18.0]),
        ),
    }

    reduced = apply_to_collection(to_reduce, (torch.Tensor, numbers.Number, np.ndarray), lambda x: x * 2)

    assert isinstance(reduced, dict), " Type Consistency of dict not preserved"
    assert all(x in reduced for x in to_reduce), "Not all entries of the dict were preserved"
    assert all(
        isinstance(reduced[k], type(expected_result[k])) for k in to_reduce
    ), "At least one type was not correctly preserved"

    assert isinstance(reduced["a"], torch.Tensor), "Reduction Result of a Tensor should be a Tensor"
    assert torch.allclose(expected_result["a"], reduced["a"]), "Reduction of a tensor does not yield the expected value"

    assert isinstance(reduced["b"], list), "Reduction Result of a list should be a list"
    assert all(
        torch.allclose(x, y) for x, y in zip(reduced["b"], expected_result["b"])
    ), "At least one value of list reduction did not come out as expected"

    assert isinstance(reduced["c"], tuple), "Reduction Result of a tuple should be a tuple"
    assert all(
        torch.allclose(x, y) for x, y in zip(reduced["c"], expected_result["c"])
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

    assert dataclasses.is_dataclass(reduced["h"]) and not isinstance(
        reduced["h"], type
    ), "Reduction of a dataclass should result in a dataclass"
    assert torch.allclose(
        reduced["h"].input_ids, expected_result["h"].input_ids
    ), "Reduction of a dataclass did not yield the desired result"
    assert np.allclose(
        reduced["h"].segment_ids, expected_result["h"].segment_ids
    ), "Reduction of a dataclass did not yield the desired result"

    assert dataclasses.is_dataclass(reduced["i"]) and not isinstance(
        reduced["i"], type
    ), "Reduction of a dataclass should result in a dataclass"
    assert dataclasses.is_dataclass(reduced["i"].feature) and not isinstance(
        reduced["i"].feature, type
    ), "Reduction of a nested dataclass should result in a nested dataclass"
    assert (
        reduced["i"].example_ids == expected_result["i"].example_ids
    ), "Reduction of a nested dataclass did not yield the desired result"
    assert torch.allclose(
        reduced["i"].label, expected_result["i"].label
    ), "Reduction of a nested dataclass did not yield the desired result"
    assert torch.allclose(
        reduced["i"].feature.input_ids, expected_result["i"].feature.input_ids
    ), "Reduction of a nested dataclass did not yield the desired result"
    assert np.allclose(
        reduced["i"].feature.segment_ids, expected_result["i"].feature.segment_ids
    ), "Reduction of a nested dataclass did not yield the desired result"

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


def test_apply_to_collection_include_none():
    to_reduce = [1, 2, 3.4, 5.6, 7, (8, 9.1, {10: 10})]

    def fn(x):
        if isinstance(x, float):
            return x

    reduced = apply_to_collection(to_reduce, (int, float), fn)
    assert reduced == [None, None, 3.4, 5.6, None, (None, 9.1, {10: None})]

    reduced = apply_to_collection(to_reduce, (int, float), fn, include_none=False)
    assert reduced == [3.4, 5.6, (9.1, {})]


def test_apply_to_collections():
    to_reduce_1 = {"a": {"b": [1, 2]}, "c": 5}
    to_reduce_2 = {"a": {"b": [3, 4]}, "c": 6}

    def fn(a, b):
        return a + b

    # basic test
    reduced = apply_to_collections(to_reduce_1, to_reduce_2, int, fn)
    assert reduced == {"a": {"b": [4, 6]}, "c": 11}

    with pytest.raises(KeyError):
        # strict mode - if a key does not exist in both we fail
        apply_to_collections({**to_reduce_2, "d": "foo"}, to_reduce_1, float, fn)

    # multiple dtypes
    reduced = apply_to_collections(to_reduce_1, to_reduce_2, (list, int), fn)
    assert reduced == {"a": {"b": [1, 2, 3, 4]}, "c": 11}

    # wrong dtype
    reduced = apply_to_collections(to_reduce_1, to_reduce_2, (list, int), fn, wrong_dtype=int)
    assert reduced == {"a": {"b": [1, 2, 3, 4]}, "c": 5}

    # list takes precedence because it is the type of data1
    reduced = apply_to_collections([1, 2, 3], [4], (int, list), fn)
    assert reduced == [1, 2, 3, 4]

    # different sizes
    with pytest.raises(AssertionError, match="Sequence collections have different sizes"):
        apply_to_collections([[1, 2], [3]], [4], int, fn)

    def fn(a, b):
        return a.keys() | b.keys()

    # base case
    reduced = apply_to_collections(to_reduce_1, to_reduce_2, dict, fn)
    assert reduced == {"a", "c"}

    # type conversion
    to_reduce = [(1, 2), (3, 4)]
    reduced = apply_to_collections(to_reduce, to_reduce, int, lambda *x: sum(x))
    assert reduced == [(2, 4), (6, 8)]

    # named tuple
    foo = namedtuple("Foo", ["bar"])
    to_reduce = [foo(1), foo(2), foo(3)]
    reduced = apply_to_collections(to_reduce, to_reduce, int, lambda *x: sum(x))
    assert reduced == [foo(2), foo(4), foo(6)]

    # passing none
    reduced1 = apply_to_collections([1, 2, 3], None, int, lambda x: x * x)
    reduced2 = apply_to_collections(None, [1, 2, 3], int, lambda x: x * x)
    assert reduced1 == reduced2 == [1, 4, 9]
    reduced = apply_to_collections(None, None, int, lambda x: x * x)
    assert reduced is None


@pytest.mark.parametrize("should_return", [False, True])
def test_wrongly_implemented_transferable_data_type(should_return):
    class TensorObject:
        def __init__(self, tensor: torch.Tensor, should_return: bool = True):
            self.tensor = tensor
            self.should_return = should_return

        def to(self, device):
            self.tensor.to(device)
            # simulate a user forgets to return self
            if self.should_return:
                return self

    tensor = torch.tensor(0.1)
    obj = TensorObject(tensor, should_return)
    assert obj == move_data_to_device(obj, torch.device("cpu"))
