import abc


class PrecisionPlugin(abc.ABC):
    """
    Abstract class to extend for precision support (32/16 etc).

    This is extended to cover any specific logic required for precision support such as AMP/APEX or sharded
    training.
    """

    def connect(self, model, optimizers):
        raise NotImplementedError

    def training_step(self, fx, args):
        raise NotImplementedError

    def backward(self, closure_loss, optimizer, opt_idx, *args, **kwargs):
        raise NotImplementedError

    def clip_gradients(self, grad_clip_val, model, optimizer):
        raise NotImplementedError
