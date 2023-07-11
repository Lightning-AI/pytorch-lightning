from itertools import chain

import typing
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)
from collections import OrderedDict

import torch
from torch import Tensor
import torch.nn as nn
from lightning.pytorch.core.module import LightningModule


class FlatParameter(nn.Parameter):
    """A parameter that is initialized from a list of parameters and can be
    turned into a list of views as needed.
    """

    def __new__(cls, params: Sequence[nn.Parameter], requires_grad: bool = True) -> "FlatParameter":
        """Make an object using the parent's __new__ function."""

        # A empty of non-list input doesn't make sense.
        if not isinstance(params, (list, tuple)) or len(params) == 0:
            raise ValueError("An non-empty list or tuple argument is needed")

        # Normally, all items are Parameters. But during pickling, we will have a single
        # Tensor as the input and later in __init__, the correct _param_numels and _param_shapes
        # are set.
        if not all(isinstance(p, (nn.Parameter, Tensor)) for p in params):
            raise ValueError("List items need to be Parameter types")

        # Flattening involves (1) making a tensor flat (i.e. single dimensional) and (2) making a module
        # hierarchy flat (using a single tensor to replace a tree of tensors). Therefore,
        # adding back nesting and hierarchy is counter-productive. If nesting is encountered
        # in the future, the reasonable thing to do is likely for the top level FlatParameter to
        # absorb the nested one and keep the result flat, free from hierarchy.
        if any(isinstance(p, FlatParameter) for p in params):
            raise ValueError("Nesting FlatParameter is not supported")

        # concatenate  all parameters into a single dimension parameter
        data = torch.cat([p.detach().reshape(-1) if isinstance(p, nn.Parameter) else p.reshape(-1) for p in params], 0)
        
        return super(FlatParameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __init__(self, params: Sequence[nn.Parameter], requires_grad: bool = True):
        """Initialize the _param_numels and _param_shapes lists."""
        self._param_numels = [p.numel() for p in params]
        
        # TODO: Verify the number of parameters when sharding model
        # assert self.numel() <= sum(
        #     self._param_numels
        # ), f"Something wrong with __new__ method, {self.numel()} vs. {sum(self._param_numels)}"
        
        self._param_shapes = [p.size() for p in params]
        # These are set by FPW class below, not by this class itself.
        self._param_infos: List[Tuple[str, nn.Module, str]] = []
        self._shared_param_infos: List[Tuple[str, str, nn.Module, str, nn.Module, str]] = []

    def get_param_views(self) -> Iterator[Tensor]:
        """Return a generator of views that map to the original parameters."""
        # Note, self.data could be sharded, so its numel is <= to the sum.
        # TODO: Figure out what this means
        assert self.data.numel() <= sum(
            self._param_numels
        ), f"Incorrect internal state {self.data.numel()} vs. {sum(self._param_numels)}"
        if self.numel() != sum(self._param_numels):
            raise ValueError(
                f"Incorrect numel of supplied data: got {self.numel()} but expected {sum(self._param_numels)}"
            )
        return (t.view(s) for (t, s) in zip(self.split(self._param_numels), self._param_shapes))


class FlattenParamsWrapper(LightningModule):
    """
    A wrapper for transparently flattening a Module's parameters.
    [1] https://github.com/SsnL/PyTorch-Reparam-Module
    [2] https://github.com/facebookresearch/fairscale/blob/main/fairscale/nn/misc/flatten_params_wrapper.py

    Args:
        module (lightning.LightningModule):
            The module to wrap.
        param_list (Optional[List[List[nn.Parameter]]]):
            Only flatten parameters appearing in the given groups.
            param_list cannot be an empty list.
            Note, if a single param is in one of the list, it still get flattened and the
            original param is removed and replaced with the flatten one.
            Default: None, flatten all parameters (if any)
    """

    def __init__(
        self,
        module: LightningModule,
        param_list: Optional[List[List[nn.Parameter]]] = None,
        # flat_param_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self._fpw_module = module
        self.is_flattened = False
        self.optimizer_is_init = False

        # Because self._fpw_module is LightningModule, all functions inherited from LightningModule must be overloaded with
        # functions from _fpw_module  
        if param_list is None:
            param_list = [list(module.parameters())]

        # Be backward compatible and turn a single param list into a list of list.
        # If input param_list is not List[List[nn.Parameter]] raise error
        if len(param_list) == 0:
            raise ValueError ("Parameter list cannot be empty")
        if type(param_list).__name__ != 'list' or type(param_list[0]).__name__ != 'list':
            raise ValueError ("Expected the parameter list to be a list of list")

        # Since the parameters will be deleted, let's record the number original
        # parameters managed by this class.
        self.num_params_managed = 0

        # List containing Set(Tuple(Module (Conv) and the name of the parameter (weight/bias)))
        # Create a seperate set for each param group
        # NOTE: Parameters cannot be shared across p_lists i.e the same parameter cannot exist in
        # more than one list, as we will create seperate optimizers for each list if needed, and 
        # a single parameter should not be optimized twice in the same iteration. Each group in 
        # _param_sets is flattened individually.
        self._param_sets = []   
        
        # Set of all flattened parameters
        overall_param_set: Set[nn.Parameter] = set()
        for p_list in param_list:
            # Remove any duplicates from the list.
            p_set: Set[nn.Parameter] = set(p_list)

            self.num_params_managed += len(p_set)
            overall_param_set = overall_param_set.union(p_set)

            # Convert from list of Parameters to set of (Module, name) tuples,
            # which will survive in case the parameter instances are reset.
            # Also, a shared param will correctly appear under multiple modules
            # as they should.

            # Set of Tuple(Module, param) of all parameters in current param_list that
            # have to be flattened
            new_p_set_with_names = set()
            for m in self.modules():
                for n, p in m.named_parameters(recurse=False): 
                    # the FPW class is a nn.Module itself, and _fpw_module is
                    # a submodule here. Recursing over the _fpw_module, we 
                    # have the layers of the model as submodules.
                    # m will be the name of the final submodule which contains the parameters
                    # such as conv layers, and as we already recurse over the entire model
                    # in the first for loop, we need not recurse when searching for params  
                    if p in p_set:
                        new_p_set_with_names.add((m, n))
            if new_p_set_with_names:
                self._param_sets.append(new_p_set_with_names)

        if len(overall_param_set) != self.num_params_managed:
            # Each p_list above could have shared params. However, you can't
            # have shared params across different p_list. That means part of
            # the flattened parameter must be shared, which is impossible to
            # support.
            raise ValueError(f"Incorrect param groups {len(overall_param_set)} vs {self.num_param_managed}, some parameters appear in more than one param group")

        # List of FlatParams() for each param group
        self.flat_params: List[nn.Parameter] = []
        # Index of each parameter in the flat param group
        self.params2idx = {}
        self.idx2params = {}

        # Prepare flat param names.
        # TODO: Only when flat_param_names is not given
        # if flat_param_names is None:
        flat_param_names = [f"{i}" for i, _ in enumerate(self._param_sets)]
        if len(flat_param_names) != len(self._param_sets):
            raise ValueError("Names and number of param lists must be equal")
        if len(flat_param_names) != len(set(flat_param_names)):
            raise ValueError("Each flat param must be given a unique name")
        self.flat_param_names = [f"flat_param_{n}" for n in flat_param_names]
        # flat_param: Optional[nn.Parameter] = None

        # Initialize all flat_params.
        # NOTE: All params may not be flattened
        for i in range(len(self._param_sets)):
            new_p_set = self._param_sets[i]
            params, param_infos, shared_param_infos = self._init_flatten_params(new_p_set)
            flat_param = FlatParameter(params, params[0].requires_grad)
            flat_param._param_infos = param_infos
            flat_param._shared_param_infos = shared_param_infos
            self.flat_params.append(flat_param)
            for p in params:
                self.params2idx[p] = i
            self.idx2params[i] = params
        # NOTE: The flattened parameters are actually set as parameters in configure_optimizers
        # so that we can create an optimizer for the flattened params

    def _init_flatten_params(
        self, p_set: Set[Tuple[nn.Module, str]]
    ) -> Tuple[
        List[nn.Parameter], List[Tuple[str, nn.Module, str]], List[Tuple[str, str, nn.Module, str, nn.Module, str]]
    ]:
        """
        Args:
            p_set (set):
                A set of (module, param_name) for a set of params that needed
                to be flattened. There could be shared params in this set.
        """
        param_infos = []
        # [(module_name='_fpw_module.0', Module=Conv2d(..), param_name=('weight'))]
        shared_param_memo: Dict[nn.Parameter, Tuple[str, nn.Module, str]] = {}
        # {param: (module_name='_fpw_module.0', Module=Conv2d(..), param_name=('weight'))}
        shared_param_infos = []
        # [(module_name1, module_name2, Module1, param_name1, Module2, param_name2)]
        params = []
        # [params]
        fp32 = []
        fp16 = []
        for module_name, m in self.named_modules():
            for n, p in m.named_parameters(recurse=False):
                if p.dtype == torch.float32:
                    fp32.append(module_name)
                elif p.dtype == torch.float16:
                    fp16.append(module_name)
                else:
                    raise ValueError (f"Parameters have different precisions")
                if p is not None and (m, n) in p_set:
                    if p in shared_param_memo:
                        # Only if a parameter is shared by two or more submodules, eg RNN weights
                        mname, shared_m, shared_n = shared_param_memo[p]
                        shared_param_infos.append((module_name, mname, m, n, shared_m, shared_n))
                    else:
                        shared_param_memo[p] = (module_name, m, n)
                        param_infos.append((module_name, m, n))
                        params.append(p)

        del shared_param_memo
        fp16_msg, fp32_msg = ",".join(fp16), ",".join(fp32)
        assert (
            len(set(p.dtype for p in params)) == 1
        ), f"expects all parameters to have same dtype: fp32: {fp32_msg} \n fp16: {fp16_msg} "
        assert (
            len(set(p.requires_grad for p in params)) == 1
        ), f"expects all parameters to have same requires_grad {p_set}"
        assert len(params) == len(set(params)), "params list should not have dups"
        return params, param_infos, shared_param_infos


    def _flatten_params(self, flat_params: List[nn.Parameter]) -> None:
        """Flatten the managed parameters and replaced the original
        attributes with views to the flat params.
        """
        assert not self.is_flattened
        self.is_flattened = True

        # Modified this as FPW has no attribute _param_infos
        # deregister the names as parameters
        # Make sure device of all tensors is the same, and push FlatTensor to the correct device
        device = None
        for param in flat_params:
            for _, m, n in param._param_infos:
                n_device = getattr(m, n).device
                if device is None:
                    device = n_device
                if device != n_device:
                    raise ValueError(f"All tensors must be on the same device, got devices {device}, {n_device}")
                delattr(m, n)
            for _, _, m, n, _, _ in param._shared_param_infos:
                n_device = getattr(m, n).device
                if device is None:
                    device = n_device
                if device != n_device:
                    raise ValueError(f"All tensors must be on the same device, got devices {device}, {n_device}")
                delattr(m, n)

        # register the flatten ones and save it to self.
        assert len(self.flat_param_names) == len(flat_params), f"{len(self.flat_param_names)} vs. {len(flat_params)}"
        for n, flat_param in zip(self.flat_param_names, flat_params):
            flat_param.data = flat_param.data.to(device)
            self.register_parameter(n, flat_param)

        # register the views as plain attributes
        self._unflatten_params_as_views()

    def _unflatten_params(self) -> None:
        """Undo flattening and create separate parameters from the already flattened
        self.flat_param
        """
        assert self.is_flattened
        self.is_flattened = False

        ps = self.get_param_views()

        # Register parameters into original model
        for fp in self.flat_params:
            for (_, m, n), p in zip(fp._param_infos, ps):
                if hasattr(m, n):
                    delattr(m, n)
                m.register_parameter(n, nn.Parameter(p))
            for (_, _, m, n, shared_m, shared_n) in fp._shared_param_infos:
                if hasattr(m, n):
                    delattr(m, n)
                m.register_parameter(n, getattr(shared_m, shared_n))

        for n in self.flat_param_names:
            # This ensures the flat params are removed from the module.
            delattr(self, n)
        self.flat_params = []

    def _unflatten_params_as_views(self) -> None:
        """Unlike ``_unflatten_params``, this function unflatten into views and keep
        self.flat_param unchanged.
        """
        assert self.is_flattened
        ps = self.get_param_views()
        param_views = []
        
        for param in self.flat_params:
            for (_, m, n), p in zip(param._param_infos, ps):
                setattr(m, n, p)  # This will set as plain attr
                param_views.append(p)

        # Save param views for easy access if anyone still wants to access
        # parameters of the module.
        setattr(self._fpw_module, "_unflattened_param_views", param_views)

        for param in self.flat_params:
            for (_, _, m, n, shared_m, shared_n) in param._shared_param_infos:
                setattr(m, n, getattr(shared_m, shared_n))
        
    def forward(self, *args, **kwargs):
        self._unflatten_params_as_views()
        return self._fpw_module(*args, **kwargs)

    def training_step(self, *args, **kwargs):
        self._unflatten_params_as_views()
        return self._fpw_module.training_step(*args, **kwargs)

    def validation_step(self , *args, **kwargs):
        self._unflatten_params_as_views()
        return self._fpw_module.training_step(*args, **kwargs)

    def test_step(self , *args, **kwargs):
        self._unflatten_params_as_views()
        return self._fpw_module.test_step(*args, **kwargs)

    def predict_step(self , *args, **kwargs):
        return self._fpw_module.predict_step(*args, **kwargs)

    def configure_callbacks(self):
        return self._fpw_module.configure_callbacks()

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        self._fpw_module.configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)

    def configure_optimizers(self, *args, **kwargs):
        # TODO: Change for multiple param groups
        # TODO: Verify multiple param groups, one where some parameter is not included in any
        # Get user defined optimizer
        optimizer = self._fpw_module.configure_optimizers(*args, **kwargs)
        optimizer_flattened = self._fpw_module.configure_optimizers(*args, **kwargs)
        
        # Handle the case of multiple optimizers by converting optimizers to a tuple
        if type(optimizer).__name__ != 'tuple':
            optimizer = (optimizer,)
            optimizer_flattened = (optimizer_flattened,)

        # Register FlatParams as parameters and delete old parameters 
        self._flatten_params(self.flat_params)

        flat_params = 0
        skipped_params = 0
        for opt_idx in range(len(optimizer)):
            for pg_idx in range(len(optimizer[opt_idx].param_groups)):
                pg = optimizer[opt_idx].param_groups[pg_idx]
                params = pg['params']
                p = params[0]
                idx = self.params2idx.get(p, None)
                if idx is not None:
                    assert self.idx2params[idx] == params, f"Flattened parameters at index {idx} and parameter group {pg_idx} do not match for optimizer {opt_idx}"
                    flat_params += len(self.idx2params[idx])
                    optimizer_flattened[opt_idx].param_groups[pg_idx]['params'] = [getattr(self, self.flat_param_names[idx])]
                else:
                    for p1 in params:
                        assert self.params2idx.get(p1, None) == idx, f"Some parameters in this group are expected to be flattened while others are not"
                        skipped_params += 1

        assert flat_params == self.num_params_managed, f"All flattened parameters (total: {self.num_params_managed}) do not appear in the optimizer (total: {flat_params})"
        assert flat_params + skipped_params == self.num_params_managed + len(list(self._fpw_module.parameters())), f"Total number parameters in the model (total: {self.num_params_managed + len(list(self._fpw_module.parameters()))}) does not match the number of parameters in all optimizers combined (total: {flat_params + skipped_params}): Some parameters are not optimized"

        del optimizer
        return optimizer_flattened

    def manual_backward(self, loss, *args, **kwargs):
        return self._fpw_module.manual_backward(loss, *args, **kwargs)

    def backward(self, loss, *args, **kwargs):
        return self._fpw_module.backward(loss, *args, **kwargs)

    def lr_scheduler_step(self, scheduler, metric):
        self._fpw_module.lr_scheduler_step(scheduler, metric)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        self._fpw_module.optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        self._fpw_module.optimizer_zero_grad(epoch, batch_idx, optimizer)

    def get_param_views(self) -> Iterator[Tensor]:
        """Used to get a generator over all views from a list of data list."""
        params = self.flat_params        
        gens = []
        for p in params:
            gens.append(p.get_param_views())  # type: ignore

        return chain(*gens)