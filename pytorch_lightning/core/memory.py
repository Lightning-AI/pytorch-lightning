"""
Generates a summary of a model's layers and dimensionality
"""

import gc
import os
import subprocess
from subprocess import PIPE
from typing import Tuple, Dict, Union, List

import numpy as np
import torch
from torch.nn import Module

import pytorch_lightning as pl

from pytorch_lightning import _logger as log


class ModelSummary(object):

    def __init__(self, model: 'pl.LightningModule', mode: str = 'full'):
        """ Generates summaries of model layers and dimensions. """
        self.model = model
        self.mode = mode
        self.in_sizes = []
        self.out_sizes = []

        self.summarize()

    def __str__(self):
        return self.summary.__str__()

    def __repr__(self):
        return self.summary.__str__()

    def named_modules(self) -> List[Tuple[str, Module]]:
        if self.mode == 'full':
            mods = self.model.named_modules()
            mods = list(mods)[1:]  # do not include root module (LightningModule)
        elif self.mode == 'top':
            # the children are the top-level modules
            mods = self.model.named_children()
        else:
            mods = []
        return list(mods)

    def get_variable_sizes(self) -> None:
        """ Run sample input through each layer to get output sizes. """
        mods = self.named_modules()
        in_sizes = []
        out_sizes = []
        input_ = self.model.example_input_array

        if self.model.on_gpu:
            device = next(self.model.parameters()).get_device()
            # test if input is a list or a tuple
            if isinstance(input_, (list, tuple)):
                input_ = [input_i.cuda(device) if torch.is_tensor(input_i) else input_i
                          for input_i in input_]
            else:
                input_ = input_.cuda(device)

        if self.model.trainer.use_amp:
            # test if it is not a list or a tuple
            if isinstance(input_, (list, tuple)):
                input_ = [input_i.half() if torch.is_tensor(input_i) else input_i
                          for input_i in input_]
            else:
                input_ = input_.half()

        with torch.no_grad():

            for _, m in mods:
                if isinstance(input_, (list, tuple)):  # pragma: no-cover
                    out = m(*input_)
                else:
                    out = m(input_)

                if isinstance(input_, (list, tuple)):  # pragma: no-cover
                    in_size = []
                    for x in input_:
                        if isinstance(x, list):
                            in_size.append(len(x))
                        else:
                            in_size.append(x.size())
                else:
                    in_size = np.array(input_.size())

                in_sizes.append(in_size)

                if isinstance(out, (list, tuple)):  # pragma: no-cover
                    out_size = np.asarray([x.size() for x in out])
                else:
                    out_size = np.array(out.size())

                out_sizes.append(out_size)
                input_ = out

        self.in_sizes = in_sizes
        self.out_sizes = out_sizes
        assert len(in_sizes) == len(out_sizes)

    def get_layer_names(self) -> None:
        """ Collect Layer Names """
        mods = self.named_modules()
        names = []
        layers = []
        for name, m in mods:
            names += [name]
            layers += [str(m.__class__)]

        layer_types = [x.split('.')[-1][:-2] for x in layers]

        self.layer_names = names
        self.layer_types = layer_types

    def get_parameter_sizes(self) -> None:
        """ Get sizes of all parameters in `model`. """
        mods = self.named_modules()
        sizes = []
        for _, m in mods:
            p = list(m.parameters())
            modsz = [np.array(param.size()) for param in p]
            sizes.append(modsz)

        self.param_sizes = sizes

    def get_parameter_nums(self) -> None:
        """ Get number of parameters in each layer. """
        param_nums = []
        for mod in self.param_sizes:
            all_params = 0
            for p in mod:
                all_params += np.prod(p)
            param_nums.append(all_params)
        self.param_nums = param_nums

    def make_summary(self) -> None:
        """
        Makes a summary listing with:

        Layer Name, Layer Type, Input Size, Output Size, Number of Parameters
        """
        arrays = [['Name', self.layer_names],
                  ['Type', self.layer_types],
                  ['Params', list(map(get_human_readable_count, self.param_nums))]]
        if self.model.example_input_array is not None:
            arrays.append(['In sizes', self.in_sizes])
            arrays.append(['Out sizes', self.out_sizes])

        self.summary = _format_summary_table(*arrays)

    def summarize(self) -> None:
        self.get_layer_names()
        self.get_parameter_sizes()
        self.get_parameter_nums()

        if self.model.example_input_array is not None:
            self.get_variable_sizes()
        self.make_summary()


def _format_summary_table(*cols) -> str:
    """
    Takes in a number of arrays, each specifying a column in
    the summary table, and combines them all into one big
    string defining the summary table that are nicely formatted.
    """
    n_rows = len(cols[0][1])
    n_cols = 1 + len(cols)

    # Layer counter
    counter = list(map(str, list(range(n_rows))))
    counter_len = max([len(c) for c in counter])

    # Get formatting length of each column
    length = []
    for c in cols:
        str_l = len(c[0])  # default length is header length
        for a in c[1]:
            if isinstance(a, np.ndarray):
                array_string = '[' + ', '.join([str(j) for j in a]) + ']'
                str_l = max(len(array_string), str_l)
            else:
                str_l = max(len(a), str_l)
        length.append(str_l)

    # Formatting
    s = '{:<{}}'
    full_length = sum(length) + 3 * n_cols
    header = [s.format(' ', counter_len)] + [s.format(c[0], l) for c, l in zip(cols, length)]

    # Summary = header + divider + Rest of table
    summary = ' | '.join(header) + '\n' + '-' * full_length
    for i in range(n_rows):
        line = s.format(counter[i], counter_len)
        for c, l in zip(cols, length):
            if isinstance(c[1][i], np.ndarray):
                array_string = '[' + ', '.join([str(j) for j in c[1][i]]) + ']'
                line += ' | ' + array_string + ' ' * (l - len(array_string))
            else:
                line += ' | ' + s.format(c[1][i], l)
        summary += '\n' + line

    return summary


def print_mem_stack() -> None:  # pragma: no-cover
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                log.info(type(obj), obj.size())
        except Exception:
            pass


def count_mem_items() -> Tuple[int, int]:  # pragma: no-cover
    num_params = 0
    num_tensors = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                obj_type = str(type(obj))
                if 'parameter' in obj_type:
                    num_params += 1
                else:
                    num_tensors += 1
        except Exception:
            pass

    return num_params, num_tensors


def get_memory_profile(mode: str) -> Union[Dict[str, int], Dict[int, int]]:
    """ Get a profile of the current memory usage.

    Args:
        mode: There are two modes:

            - 'all' means return memory for all gpus
            - 'min_max' means return memory for max and min

    Return:
        A dictionary in which the keys are device ids as integers and
        values are memory usage as integers in MB.
        If mode is 'min_max', the dictionary will also contain two additional keys:

        - 'min_gpu_mem': the minimum memory usage in MB
        - 'max_gpu_mem': the maximum memory usage in MB
    """
    memory_map = get_gpu_memory_map()

    if mode == 'min_max':
        min_index, min_memory = min(memory_map.items(), key=lambda item: item[1])
        max_index, max_memory = max(memory_map.items(), key=lambda item: item[1])

        memory_map = {'min_gpu_mem': min_memory, 'max_gpu_mem': max_memory}

    return memory_map


def get_gpu_memory_map() -> Dict[str, int]:
    """Get the current gpu usage.

    Return:
        A dictionary in which the keys are device ids as integers and
        values are memory usage as integers in MB.
    """
    result = subprocess.run(
        [
            'nvidia-smi',
            '--query-gpu=memory.used',
            '--format=csv,nounits,noheader',
        ],
        encoding='utf-8',
        # capture_output=True,          # valid for python version >=3.7
        stdout=PIPE, stderr=PIPE,       # for backward compatibility with python version 3.6
        check=True)
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.stdout.strip().split(os.linesep)]
    gpu_memory_map = {f'gpu_{index}': memory for index, memory in enumerate(gpu_memory)}
    return gpu_memory_map


def get_human_readable_count(number: int) -> str:
    """
    Abbreviates an integer number with K, M, B, T for thousands, millions,
    billions and trillions, respectively.

    Examples:
        >>> get_human_readable_count(123)
        '123  '
        >>> get_human_readable_count(1234)  # (one thousand)
        '1 K'
        >>> get_human_readable_count(2e6)   # (two million)
        '2 M'
        >>> get_human_readable_count(3e9)   # (three billion)
        '3 B'
        >>> get_human_readable_count(4e12)  # (four trillion)
        '4 T'
        >>> get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'

    Args:
        number: a positive integer number

    Return:
        A string formatted according to the pattern described above.

    """
    assert number >= 0
    labels = [' ', 'K', 'M', 'B', 'T']
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10 ** shift)
    index = num_groups - 1
    return f'{int(number):,d} {labels[index]}'
