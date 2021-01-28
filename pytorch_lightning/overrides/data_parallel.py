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
import warnings
from collections.abc import Iterable, Mapping
from itertools import chain
from typing import Any, Optional

import torch
from torch import Tensor
from torch.cuda._utils import _get_device_index
from torch.nn import DataParallel, Module
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel._functions import Gather

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.warnings import WarningCache


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


warning_cache = WarningCache()


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

            running_stage = self.module.running_stage

            if running_stage == RunningStage.TRAINING:
                return self.module.training_step(*inputs[0], **kwargs[0])

            elif running_stage == RunningStage.TESTING:
                return self.module.test_step(*inputs[0], **kwargs[0])

            elif running_stage == RunningStage.EVALUATING:
                return self.module.validation_step(*inputs[0], **kwargs[0])

            else:
                return self.module.predict(*inputs[0], **kwargs[0], skip_collate_fn=True)

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

    def __init__(self, module: LightningModule, *args, **kwargs):
        warnings.warn(
            "The usage of `LightningDistributedDataParallel` is deprecated since v1.2 and will be removed in v1.4."
            " From now on we recommend to directly sublcass `torch.nn.parallel.DistributedDataParallel`.",
            DeprecationWarning
        )
        super().__init__(LightningDistributedModule(module), *args, **kwargs)


class LightningDistributedModule(torch.nn.Module):

    def __init__(self, pl_module: LightningModule):
        """
        Wraps the user's LightningModule and redirects the forward call to the appropriate
        method, either ``training_step``, ``validation_step`` or ```test_step``.
        This class is used in combination with :class:`~torch.nn.parallel.DistributedDataParallel` as
        shown in the example.

        Example:

            ddp_model = DistributedDataParallel(
                module=LightningDistributedModule(lightning_module),
                device_ids=[local_rank],
                ...
            )

        Args:
            pl_module: the model to wrap

        """
        super().__init__()
        self.module = pl_module

    def forward(self, *inputs, **kwargs):

        running_stage = self.module.running_stage

        if running_stage == RunningStage.TRAINING:
            output = self.module.training_step(*inputs, **kwargs)
            warn_if_output_is_none(output, "training_step")

        elif running_stage == RunningStage.TESTING:
            output = self.module.test_step(*inputs, **kwargs)
            warn_if_output_is_none(output, "test_step")

        elif running_stage == RunningStage.EVALUATING:
            output = self.module.validation_step(*inputs, **kwargs)
            warn_if_output_is_none(output, "validation_step")

        else:
            output = self.module.predict(*inputs, **kwargs, skip_collate_fn=True)

        return output


# In manual_optimization, we need to call reducer prepare_for_backward.
# Note: Keep track of Pytorch DDP and update if there is a change
# https://github.com/pytorch/pytorch/blob/v1.7.1/torch/nn/parallel/distributed.py#L626-L638
def prepare_for_backward(model: DistributedDataParallel, output: Any):
    if torch.is_grad_enabled() and model.require_backward_grad_sync:
        model.require_forward_param_sync = True
        # We'll return the output object verbatim since it is a freeform
        # object. We need to find any tensors in this object, though,
        # because we need to figure out which parameters were used during
        # this forward pass, to ensure we short circuit reduction for any
        # unused parameters. Only if `find_unused_parameters` is set.
        if model.find_unused_parameters:
            model.reducer.prepare_for_backward(list(_find_tensors(output)))
        else:
            model.reducer.prepare_for_backward([])
    else:
        model.require_forward_param_sync = False


def warn_if_output_is_none(output: Any, method_name: str) -> None:
    if output is None:
        warning_cache.warn(f'Your {method_name} returned None. Did you forget to return an output?')


def warn_missing_output(fx_called):
    if fx_called == 'training_step':
        warning_cache.warn("Your training_step returned None. Make sure that was your intention!")


def parallel_apply(
        modules: Module,
        inputs: Tensor,
        kwargs_tup: Optional[tuple] = None,
        devices: Optional[list] = None,
):  # pragma: no-cover
    r"""Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.

    Args:
        modules: modules to be parallelized
        inputs: inputs to the modules
        devices: CUDA devices

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
                if module.running_stage == RunningStage.TRAINING:
                    output = module.training_step(*input, **kwargs)
                    fx_called = 'training_step'

                elif module.running_stage == RunningStage.TESTING:
                    output = module.test_step(*input, **kwargs)
                    fx_called = 'test_step'

                elif module.running_stage == RunningStage.EVALUATING:
                    output = module.validation_step(*input, **kwargs)
                    fx_called = 'validation_step'

                else:
                    output = module.predict(*input, **kwargs, skip_collate_fn=True)
                    fx_called = 'predict'

                if output is None:
                    warn_missing_output(fx_called)

                if output is not None and module._distrib_type in ('dp', 'ddp2'):
                    auto_squeeze_dim_zeros(output)
                # ---------------

            with lock:
                results[i] = output
        # todo: specify the possible exception
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
    if isinstance(output, torch.Tensor):
        output = output.unsqueeze(0)
        return output

    for k, v in output.items():
        if not isinstance(v, torch.Tensor):
            continue

        is_scalar = v.dim() == 0
        if is_scalar:
            output[k] = output[k].unsqueeze(0)
