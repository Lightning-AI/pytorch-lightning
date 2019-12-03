import inspect
from abc import ABC, abstractmethod

from pytorch_lightning.core.lightning import LightningModule


class TrainerModelHooksMixin(ABC):

    def is_function_implemented(self, f_name):
        model = self.get_model()
        f_op = getattr(model, f_name, None)
        return callable(f_op)

    def is_overriden(self, f_name):
        model = self.get_model()
        super_object = LightningModule

        # when code pointers are different, it was overriden
        is_overriden = getattr(model, f_name).__code__ is not getattr(super_object, f_name).__code__
        return is_overriden

    def has_arg(self, f_name, arg_name):
        model = self.get_model()
        f_op = getattr(model, f_name, None)
        return arg_name in inspect.signature(f_op).parameters

    @abstractmethod
    def get_model(self):
        # this is just empty shell for code from other class
        pass
