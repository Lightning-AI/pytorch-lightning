from torch import nn
from torch import optim


class OptimizerConfig(nn.Module):

    def choose_optimizer(self, optimizer, params, optimizer_params, opt_name_key):
        if optimizer == 'adam':
            optimizer = optim.Adam(params, **optimizer_params)
        if optimizer == 'sparse_adam':
            optimizer = optim.SparseAdam(params, **optimizer_params)
        if optimizer == 'sgd':
            optimizer = optim.SGD(params, **optimizer_params)
        if optimizer == 'adadelta':
            optimizer = optim.Adadelta(params, **optimizer_params)

        # transfer opt state if loaded
        if opt_name_key in self.loaded_optimizer_states_dict:
            state = self.loaded_optimizer_states_dict[opt_name_key]
            optimizer.load_state_dict(state)

        return optimizer
