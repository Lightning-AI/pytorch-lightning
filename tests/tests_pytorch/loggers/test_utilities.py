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

import pytest

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers.utilities import _ListMap, _version


def test_version(tmp_path):
    """Verify versions of loggers are concatenated properly."""
    logger1 = CSVLogger(tmp_path, version=0)
    logger2 = CSVLogger(tmp_path, version=2)
    logger3 = CSVLogger(tmp_path, version=1)
    logger4 = CSVLogger(tmp_path, version=0)
    loggers = [logger1, logger2, logger3, logger4]
    version = _version([])
    assert version == ""
    version = _version([logger3])
    assert version == 1
    version = _version(loggers)
    assert version == "0_2_1"
    version = _version(loggers, "-")
    assert version == "0-2-1"


@pytest.mark.parametrize(
    "args",
    [
        (1, 2),
        [1, 2],
        {1, 2},
        range(2),
        _ListMap({"a": 1, "b": 2}),
    ],
)
def test_listmap_init(args):
    """Test initialization with different iterable types."""
    lm = _ListMap(args)
    assert len(lm) == len(args)
    assert isinstance(lm, list)
    if isinstance(args, _ListMap):
        assert lm == args


def test_listmap_append():
    """Test appending loggers to the collection."""
    lm = _ListMap()
    lm.append(1)
    assert len(lm) == 1
    lm.append(2)
    assert len(lm) == 2


def test_listmap_extend():
    # extent
    lm = _ListMap([1, 2])
    lm.extend([1, 2, 3])
    assert len(lm) == 5
    assert lm == [1, 2, 1, 2, 3]


def test_listmap_pop():
    lm = _ListMap([1, 2, 3, 4])
    item = lm.pop()
    assert item == 4
    assert len(lm) == 3
    item = lm.pop(1)
    assert item == 2
    assert len(lm) == 2
    assert lm == [1, 3]


def test_listmap_getitem():
    """Test getting items from the collection."""
    lm = _ListMap([1, 2])
    assert lm[0] == 1
    assert lm[1] == 2
    assert lm[-1] == 2
    assert lm[0:2] == [1, 2]


def test_listmap_setitem():
    """Test setting items in the collection."""
    lm = _ListMap([1, 2, 3])
    lm[0] = 10
    assert lm == [10, 2, 3]
    lm[1:3] = [20, 30]
    assert lm == [10, 20, 30]


def test_listmap_add():
    """Test adding two collections together."""
    lm1 = _ListMap([1, 2])
    lm2 = _ListMap({"3": 3, "5": 5})
    combined = lm1 + lm2
    assert isinstance(combined, _ListMap)
    assert len(combined) == 4
    assert combined is not lm1
    assert combined == [1, 2, 3, 5]
    assert combined["3"] == 3
    assert combined["5"] == 5

    ori_lm1_id = id(lm1)

    lm1 += lm2
    assert ori_lm1_id == id(lm1)
    assert isinstance(lm1, _ListMap)
    assert len(lm1) == 4
    assert lm1 == [1, 2, 3, 5]
    assert lm1["3"] == 3
    assert lm1["5"] == 5


def test_listmap_remove():
    """Test removing items from the collection."""
    lm = _ListMap([1, 2, 3])
    lm.remove(2)
    assert len(lm) == 2
    assert 2 not in lm


def test_listmap_reverse():
    """Test reversing the collection."""
    lm = _ListMap({"1": 1, "2": 2, "3": 3})
    lm.reverse()
    assert lm == [3, 2, 1]
    for (key, value), expected in zip(lm.items(), [("1", 1), ("2", 2), ("3", 3)]):
        assert (key, value) == expected


def test_listmap_reversed():
    """Test reversed iterator of the collection."""
    lm = _ListMap({"1": 1, "2": 2, "3": 3})
    rev_lm = list(reversed(lm))
    assert rev_lm == [3, 2, 1]


def test_listmap_clear():
    """Test clearing the collection."""
    lm = _ListMap({"1": 1, "2": 2, "3": 3})
    lm.clear()
    assert len(lm) == 0
    assert len(lm.keys()) == 0


def test_listmap_delitem():
    """Test deleting items from the collection."""
    lm = _ListMap({"a": 1, "b": 2, "c": 3})
    lm.extend([3, 4, 5])
    del lm["b"]
    assert len(lm) == 5
    assert "b" not in lm
    del lm[0]
    assert len(lm) == 4
    assert "a" not in lm
    assert lm == [3, 3, 4, 5]

    del lm[-1]
    assert len(lm) == 3
    assert lm == [3, 3, 4]

    del lm[-2:]
    assert len(lm) == 1
    assert lm == [3]


# Dict type properties tests
def test_listmap_keys():
    lm = _ListMap({
        "a": 1,
        "b": 2,
        "c": 3,
    })
    keys = lm.keys()
    assert set(keys) == {"a", "b", "c"}
    assert "a" in lm
    assert "d" not in lm


def test_listmap_values():
    lm = _ListMap({
        "a": 1,
        "b": 2,
        "c": 3,
    })
    values = lm.values()
    assert set(values) == {1, 2, 3}


def test_listmap_dict_items():
    lm = _ListMap({
        "a": 1,
        "b": 2,
        "c": 3,
    })
    items = lm.items()
    assert set(items) == {("a", 1), ("b", 2), ("c", 3)}


def test_listmap_dict_pop():
    lm = _ListMap({
        "a": 1,
        "b": 2,
        "c": 3,
    })
    value = lm.pop("b")
    assert value == 2
    assert "b" not in lm
    assert len(lm) == 2

    value = lm.pop(0)
    assert value == 1
    assert lm["c"] == 3  # still accessible by key
    assert len(lm) == 1
    with pytest.raises(KeyError):
        lm["a"]  # "a" was removed


def test_listmap_dict_setitem():
    lm = _ListMap({
        "a": 1,
        "b": 2,
    })
    lm["b"] = 20
    assert lm["b"] == 20
    lm["c"] = 3
    assert lm["c"] == 3
    assert len(lm) == 3


def test_listmap_sort():
    lm = _ListMap({"b": 1, "c": 3, "a": 2, "z": -7})

    lm.extend([-1, -2, 5])
    lm.sort(key=lambda x: abs(x))
    assert lm == [1, -1, 2, -2, 3, 5, -7]
    assert lm["a"] == 2
    assert lm["b"] == 1
    assert lm["c"] == 3
    assert lm["z"] == -7

    lm = _ListMap({"b": 1, "c": 3, "a": 2, "z": -7})
    lm.sort(reverse=True)
    assert lm == [3, 2, 1, -7]
    assert lm["a"] == 2
    assert lm["b"] == 1
    assert lm["c"] == 3
    assert lm["z"] == -7


def test_listmap_get():
    lm = _ListMap({"a": 1, "b": 2, "c": 3})
    assert lm.get("b") == 2
    assert lm.get("d") is None
    assert lm.get("d", 10) == 10


def test_listmap_setitem_append():
    lm = _ListMap({"a": 1, "b": 2})
    lm.append(3)
    lm["c"] = 3

    assert lm == [1, 2, 3, 3]
    assert lm["c"] == 3

    lm.remove(3)
    assert lm == [1, 2, 3]
    assert lm["c"] == 3

    lm.remove(3)
    assert lm == [1, 2]
    with pytest.raises(KeyError):
        lm["c"]  # "c" was removed
