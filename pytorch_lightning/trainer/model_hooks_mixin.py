from pytorch_lightning.root_module.root_module import LightningModule


class TrainerModelHooksMixin(object):

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
