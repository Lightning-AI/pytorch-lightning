import itertools
import threading
from itertools import chain

import torch
from torch.cuda._utils import _get_device_index
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.replicate import replicate


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


class DistributedState():
    def merge_state(self, parameter_list):
        pass

# TODO: define state container class
# TODO: maintain n separate copies in data parallel
# TODO: at end of data parallel, create a method to merge/ default opt: save from replica[0]
class LightningDataParallel(DataParallel):
    """
    CHANGE:
        add init and call super() to allow for instance var
        self.module_copies, which maintains state
    """
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(LightningDataParallel, self).__init__(
            module=module,
            device_ids=device_ids,
            output_device=output_device,
            dim=dim
        )

        #setattr(self.module, 'distributed_state', StateContainer())

        # make initial copy 
        # this gives an idea about the keys that will be replicated 
        # from the network at each forward
        self.module_copies = self.replicate(self.module, self.device_ids)

        # assign None to all key value
        # initially self.module_copies will have same keys as replicas
        for module in self.module_copies:
            for key in module.__dict__.keys():
                module.__dict__[key] = None

    def forward(self, *inputs, **kwargs):
        """
            Override the forward call in lightning so it goes to 
            training and validation step respectively
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

        # ---------------
        # CHANGE

        # combine previous state and replicas here
        # 
        # replica count is based on len(input) (<= self.module_copies)
        # replica made from initial network,
        # prev state should contain >= keys depending on prev forward
        for idx, replica in enumerate(replicas):
            for key in replica.__dict__.keys():
                # TODO: assert keys from replica are None in state?
                # add references from the replicated modules, buffers, params etc.
                self.module_copies[idx].__dict__[key] = replica.__dict__[key]

        # send in self.module_copies
        # checked backprop, works
        outputs = self.parallel_apply(self.module_copies[:len(inputs)], inputs, kwargs)

        # flush keys that were replicated from init. network
        # additional keys in self.module_copies will be retained
        # 
        # replica count is according to input (<= self.module_copies)
        for idx, replica in enumerate(replicas):
            for key in replica.__dict__.keys():
                self.module_copies[idx].__dict__[key] = None

        # CHANGE END
        # ---------------

        return self.gather(outputs, self.output_device)

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

# TODO: remove this, did not need to break replicate in the end,
# solution while maintaing atomicity of replicate will scale across 1.3.0-1.5.0
"""
def replicate(network, devices, detach=False):
    if not _replicatable_module(network):
        raise RuntimeError("Cannot replicate network where python modules are "
                           "childrens of ScriptModule")

    devices = list(map(lambda x: _get_device_index(x, True), devices))
    num_replicas = len(devices)

    params = list(network.parameters())
    param_indices = {param: idx for idx, param in enumerate(params)}
    param_copies = _broadcast_coalesced_reshape(params, devices, detach)

    buffers = list(network.buffers())
    buffers_rg = []
    buffers_not_rg = []
    for buf in buffers:
        if buf.requires_grad and not detach:
            buffers_rg.append(buf)
        else:
            buffers_not_rg.append(buf)

    buffer_indices_rg = {buf: idx for idx, buf in enumerate(buffers_rg)}
    buffer_indices_not_rg = {buf: idx for idx, buf in enumerate(buffers_not_rg)}

    buffer_copies_rg = _broadcast_coalesced_reshape(buffers_rg, devices, detach=detach)
    buffer_copies_not_rg = _broadcast_coalesced_reshape(buffers_not_rg, devices, detach=True)

    modules = list(network.modules())
    module_copies = [[] for device in devices]
    module_indices = {}
    scriptmodule_skip_attr = {"_parameters", "_buffers", "_modules", "forward", "_c"}

    for i, module in enumerate(modules):
        module_indices[module] = i
        for j in range(num_replicas):
            if _is_script_module(module):
                # we have to initialize ScriptModule properly so that
                # it works with pybind11
                def init_fn(script_module):
                    # Don't do anything here, we'll initialize the ScriptModule below
                    return
                replica = torch.jit.RecursiveScriptModule._construct(module._c._replicate_for_data_parallel(), init_fn)
            else:
                replica = module._replicate_for_data_parallel()

            module_copies[j].append(replica)

    for i, module in enumerate(modules):
        for key, child in module._modules.items():
            if child is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._modules[key] = None
            else:
                module_idx = module_indices[child]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    setattr(replica, key, module_copies[j][module_idx])
        for key, param in module._parameters.items():
            if param is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._parameters[key] = None
            else:
                param_idx = param_indices[param]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    param = param_copies[j][param_idx]
                    # parameters in replicas are no longer leaves, so remove them from _parameters
                    # and setattr them as non-parameter attributes
                    # scripted modules don't allow deleting parameters, but also don't complain
                    # on assigning non-Parameter type
                    if (not _is_script_module(replica)):
                        del replica._parameters[key]
                    setattr(replica, key, param)
        for key, buf in module._buffers.items():
            if buf is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._buffers[key] = None
            else:
                if buf.requires_grad and not detach:
                    buffer_copies = buffer_copies_rg
                    buffer_idx = buffer_indices_rg[buf]
                else:
                    buffer_copies = buffer_copies_not_rg
                    buffer_idx = buffer_indices_not_rg[buf]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    setattr(replica, key, buffer_copies[j][buffer_idx])

    return [module_copies[j][0] for j in range(num_replicas)]
"""
