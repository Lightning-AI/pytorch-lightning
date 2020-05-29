import numbers
from collections import namedtuple

import numpy as np
import torch

from pytorch_lightning.utilities.apply_func import apply_to_collection


def test_recursive_application_to_collection():
    ntc = namedtuple('Foo', ['bar'])

    to_reduce = {
        'a': torch.tensor([1.]),  # Tensor
        'b': [torch.tensor([2.])],  # list
        'c': (torch.tensor([100.]),),  # tuple
        'd': ntc(bar=5.),  # named tuple
        'e': np.array([10.]),  # numpy array
        'f': 'this_is_a_dummy_str',  # string
        'g': 12.  # number
    }

    expected_result = {
        'a': torch.tensor([2.]),
        'b': [torch.tensor([4.])],
        'c': (torch.tensor([200.]),),
        'd': ntc(bar=torch.tensor([10.])),
        'e': np.array([20.]),
        'f': 'this_is_a_dummy_str',
        'g': 24.
    }

    reduced = apply_to_collection(to_reduce, (torch.Tensor, numbers.Number, np.ndarray),
                                  lambda x: x * 2)

    assert isinstance(reduced, dict), ' Type Consistency of dict not preserved'
    assert all([x in reduced for x in to_reduce.keys()]), 'Not all entries of the dict were preserved'
    assert all([isinstance(reduced[k], type(expected_result[k])) for k in to_reduce.keys()]), \
        'At least one type was not correctly preserved'

    assert isinstance(reduced['a'], torch.Tensor), 'Reduction Result of a Tensor should be a Tensor'
    assert torch.allclose(expected_result['a'], reduced['a']), \
        'Reduction of a tensor does not yield the expected value'

    assert isinstance(reduced['b'], list), 'Reduction Result of a list should be a list'
    assert all([torch.allclose(x, y) for x, y in zip(reduced['b'], expected_result['b'])]), \
        'At least one value of list reduction did not come out as expected'

    assert isinstance(reduced['c'], tuple), 'Reduction Result of a tuple should be a tuple'
    assert all([torch.allclose(x, y) for x, y in zip(reduced['c'], expected_result['c'])]), \
        'At least one value of tuple reduction did not come out as expected'

    assert isinstance(reduced['d'], ntc), 'Type Consistency for named tuple not given'
    assert isinstance(reduced['d'].bar, numbers.Number), \
        'Failure in type promotion while reducing fields of named tuples'
    assert reduced['d'].bar == expected_result['d'].bar

    assert isinstance(reduced['e'], np.ndarray), 'Type Promotion in reduction of numpy arrays failed'
    assert reduced['e'] == expected_result['e'], \
        'Reduction of numpy array did not yield the expected result'

    assert isinstance(reduced['f'], str), 'A string should not be reduced'
    assert reduced['f'] == expected_result['f'], 'String not preserved during reduction'

    assert isinstance(reduced['g'], numbers.Number), 'Reduction of a number should result in a tensor'
    assert reduced['g'] == expected_result['g'], 'Reduction of a number did not yield the desired result'
