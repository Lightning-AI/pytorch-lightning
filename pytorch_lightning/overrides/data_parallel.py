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

import itertools
import threading
from collections import Mapping, Iterable
from itertools import chain

import torch
from torch.cuda._utils import _get_device_index
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel._functions import Gather

from pytorch_lightning.core.step_result import Result


def _find_tensors(obj):  # pragma: no-cover
    r"""
    Recursively find all tensors contained in the specified object.
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain(*map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain(*map(_find_tensors, obj.values()))
    return []


def get_a_var(obj):  # pragma: no-cover
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, (list, tuple)):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None


class LightningDataParallel(DataParallel):
    """
    Override the forward call in lightning so it goes to training and validation step respectively
    """

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            # lightning
            if self.module.training:
                return self.module.training_step(*inputs[0], **kwargs[0])
            if self.module.testing:
                return self.module.test_step(*inputs[0], **kwargs[0])

            return self.module.validation_step(*inputs[0], **kwargs[0])

        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)

        if isinstance(outputs[0], Result):
            outputs = self.__gather_structured_result(outputs)
        else:
            outputs = self.gather(outputs)
        return outputs

    def __gather_structured_result(self, outputs):
        prototype_output = outputs[0]
        original_class = prototype_output.__class__
        outputs = [dict(x) for x in outputs]

        # remove all the meta info
        meta = outputs[0]['meta']
        for i, output in enumerate(outputs):
            del output['meta']

        outputs = self.gather(outputs)

        # pass minimize to constructor for TrainResult
        if 'minimize' in outputs:
            result = original_class(outputs['minimize'])
        else:
            result = original_class()

        result.update(outputs)
        result['meta'] = meta
        return result

    def gather(self, outputs):
        r"""
        Override the gather method to support python scalars as well.
        """
        def gather_map(outputs):
            elem = outputs[0]
            elem_type = type(elem)

            if isinstance(elem, torch.Tensor):
                return Gather.apply(self.output_device, self.dim, *outputs)

            if elem is None:
                return None

            if isinstance(elem, Mapping):
                if not all((len(elem) == len(d) for d in outputs)):
                    raise ValueError('All dicts must have the same number of keys')
                return elem_type(((k, gather_map([d[k] for d in outputs]))
                                  for k in elem))

            if isinstance(elem, Iterable) and not isinstance(elem, str):
                return elem_type(map(gather_map, zip(*outputs)))

            return outputs

        # Recursive function calls like this create reference cycles.
        # Setting the function to None clears the refcycle.
        try:
            res = gather_map(outputs)
        finally:
            gather_map = None
        return res

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])


class LightningDistributedDataParallel(DistributedDataParallel):
    """
    Override the forward call in lightning so it goes to training and validation step respectively
    """

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def forward(self, *inputs, **kwargs):  # pragma: no-cover
        self._sync_params()
        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                # --------------
                # LIGHTNING MOD
                # --------------
                # normal
                # output = self.module(*inputs[0], **kwargs[0])
                # lightning
                if self.module.training:
                    output = self.module.training_step(*inputs[0], **kwargs[0])
                elif self.module.testing:
                    output = self.module.test_step(*inputs[0], **kwargs[0])
                else:
                    output = self.module.validation_step(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            # output = self.module(*inputs, **kwargs)
            # normal lightning (ddp_cpu)
            if self.module.training:
                output = self.module.training_step(*inputs, **kwargs)
            elif self.module.testing:
                output = self.module.test_step(*inputs, **kwargs)
            else:
                output = self.module.validation_step(*inputs, **kwargs)

        if torch.is_grad_enabled():
            # We'll return the output object verbatim since it is a freeform
            # object. We need to find any tensors in this object, though,
            # because we need to figure out which parameters were used during
            # this forward pass, to ensure we short circuit reduction for any
            # unused parameters. Only if `find_unused_parameters` is set.
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        return output


def parallel_apply(modules, inputs, kwargs_tup=None, devices=None):  # pragma: no-cover
    r"""Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.

    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices

    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    devices = list(map(lambda x: _get_device_index(x, True), devices))
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)

                module = module.to(device)

                # ---------------
                # CHANGE
                if module.training:
                    output = module.training_step(*input, **kwargs)

                elif module.testing:
                    output = module.test_step(*input, **kwargs)

                else:
                    output = module.validation_step(*input, **kwargs)

                if module.use_dp or module.use_ddp2:
                    auto_squeeze_dim_zeros(output)
                # ---------------

            with lock:
                results[i] = output
        except Exception as ex:
            with lock:
                results[i] = ex

    # TODO: fix hack (maybe not a hack)
    # make sure each module knows what training state it's in...
    # fixes weird bug where copies are out of sync
    root_m = modules[0]
    for m in modules[1:]:
        m.training = root_m.training
        m.testing = root_m.testing

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device))
                   for i, (module, input, kwargs, device) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


def auto_squeeze_dim_zeros(output):
    """
    In DP or DDP2 we need to unsqueeze dim 0
    :param output:
    :return:
    """
    for k, v in output.items():
        if not isinstance(v, torch.Tensor):
            continue

        is_scalar = v.dim() == 0
        if is_scalar:
            output[k] = output[k].unsqueeze(0)
