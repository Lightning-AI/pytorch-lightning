from pytorch_lightning.callbacks import Callback


class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")



def my_callbacks_factory():
    return [MyPrintingCallback()]