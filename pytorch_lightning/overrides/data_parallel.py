import itertools
import threading
from itertools import chain
from copy import deepcopy

import torch
from torch.cuda._utils import _get_device_index
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.replicate import _broadcast_coalesced_reshape
from pytorch_lightning.core import LightningModule


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
    r"""Overrides DataParallel's `forward` and `parallel_apply` methods.

    .. warning::
        It is recommended to use :class:`~torch.nn.parallel.DistributedDataParallel`,
        instead of this class, to do multi-GPU training, even if there is only a single
        node. See: :ref:`cuda-nn-ddp-instead` and :ref:`ddp`.

    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel but some types are specially handled. tensors will be
    **scattered** on dim specified (default 0). tuple, list and dict types will
    be shallow copied. The other types will be shared among different threads
    and can be corrupted if written to in the model's forward pass.

    The parallelized :attr:`module` must have its parameters and buffers on
    ``device_ids[0]`` before running this :class:`~torch.nn.DataParallel`
    module.

    .. note::
        :class:`~LightningDataParallel` adds state maintenance to :class:`~DataParallel`,
        an issue which is explained in the next warning. A container attribute
        :attr:`distributed_state` is added to :class:`~LightningModule` and a persistent
        copy of this state is stored in :attr:`distributed_buffer` for all replicas.
        At the beginning of each forward pass, the replicas are populated with state
        contained within the :attr:`distributed_buffer`. Also, the :attr:`distributed_state`
        of :attr:`module` and the replica on ``device[0]`` points to the same copy of state at
        :attr:`distributed_buffer[0]`. This implies that any changes made to variables in
        :attr:`self.distributed_state` within :class:`~LightningModule` will also be
        reflected in the :attr:`distributed_state` of the replica on ``device[0]``.

    .. warning::
        In each forward, :attr:`module` is **replicated** on each device, so any
        updates to the running module in ``forward`` will be lost. For example,
        if :attr:`module` has a counter attribute that is incremented in each
        ``forward``, it will always stay at the initial value because the update
        is done on the replicas which are destroyed after ``forward``. However,
        :class:`~torch.nn.DataParallel` guarantees that the replica on
        ``device[0]`` will have its parameters and buffers sharing storage with
        the base parallelized :attr:`module`. So **in-place** updates to the
        parameters or buffers on ``device[0]`` will be recorded. E.g.,
        :class:`~torch.nn.BatchNorm2d` and :func:`~torch.nn.utils.spectral_norm`
        rely on this behavior to update the buffers.

    .. warning::
        Forward and backward hooks defined on :attr:`module` and its submodules
        will be invoked ``len(device_ids)`` times, each with inputs located on
        a particular device. Particularly, the hooks are only guaranteed to be
        executed in correct order with respect to operations on corresponding
        devices. For example, it is not guaranteed that hooks set via
        :meth:`~torch.nn.Module.register_forward_pre_hook` be executed before
        `all` ``len(device_ids)`` :meth:`~torch.nn.Module.forward` calls, but
        that each such hook be executed before the corresponding
        :meth:`~torch.nn.Module.forward` call of that device.

    .. warning::
        When :attr:`module` returns a scalar (i.e., 0-dimensional tensor) in
        :func:`forward`, this wrapper will return a vector of length equal to
        number of devices used in data parallelism, containing the result from
        each device.

    .. note::
        There is a subtlety in using the
        ``pack sequence -> recurrent network -> unpack sequence`` pattern in a
        :class:`~torch.nn.Module` wrapped in :class:`~torch.nn.DataParallel`.
        See :ref:`pack-rnn-unpack-with-data-parallelism` section in FAQ for
        details.


    Args:
        module (Module): module to be parallelized
        device_ids (list of int or torch.device): CUDA devices (default: all devices)
        output_device (int or torch.device): device location of output (default: device_ids[0])

    Attributes:
        module (Module): the module to be parallelized
    """

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(LightningDataParallel, self).__init__(
            module=module,
            device_ids=device_ids,
            output_device=output_device,
            dim=dim
        )

        # maintains copies of state for each GPU
        self.distributed_buffer = []

        # self.distributed_buffer[0] contains a reference to self.module.distributed_state
        self.distributed_buffer.append(self.module.distributed_state)

    def forward(self, *inputs, **kwargs):
        """
        Override the forward call in lightning so it goes to training and validation step respectively
        """
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

        # scatter distributed state from buffer to replicas
        self.scatter_distributed_state(replicas, len(inputs))
        self.module.on_after_model_replicate(
            replicas, self.distributed_buffer, self.device_ids[:len(inputs)]
        )

        outputs = self.parallel_apply(replicas, inputs, kwargs)
        self.module.on_after_dp_parallel_apply(
            replicas, self.distributed_buffer, self.device_ids[:len(inputs)]
        )

        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def scatter_distributed_state(self, replicas, num_inputs):

        if len(self.distributed_buffer) == len(replicas):
            for idx in range(len(replicas)):

                buffer = self.distributed_buffer[idx]
                for var, state in buffer.__dict__.items():
                    if isinstance(state, torch.Tensor):
                        buffer.__dict__[var] = state.detach()

                replicas[idx].distributed_state = self.distributed_buffer[idx]
        else:
            state_idx = 0
            distributed_state_list = []
            distributed_state_idx = {}

            module_state_dict = self.module.distributed_state.__dict__

            for var in module_state_dict:
                if isinstance(module_state_dict[var], torch.Tensor):
                    module_state_dict[var] = module_state_dict[var].to(
                        self.module.device
                    )

                    distributed_state_list.append(module_state_dict[var])
                    distributed_state_idx[var] = state_idx

                    state_idx += 1

            distributed_state_copies = _broadcast_coalesced_reshape(
                distributed_state_list,
                self.device_ids[:num_inputs],
                detach=True
            )

            for idx in range(1, len(replicas)):
                self.distributed_buffer.append(
                    deepcopy(replicas[idx].distributed_state)
                )

                dist_buffer_dict = self.distributed_buffer[idx].__dict__

                for var in dist_buffer_dict:
                    if dist_buffer_dict[var] is not None:
                        dist_buffer_dict[var] = deepcopy(
                            distributed_state_copies[idx][distributed_state_idx[var]]
                        )

                replicas[idx].distributed_state = self.distributed_buffer[idx]

            # reference self.module.distributed_state from the 0-th replica
            replicas[0].distributed_state = self.distributed_buffer[0]


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
            # normal
            # output = self.module(*inputs, **kwargs)
            # lightning (ddp_cpu)
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
