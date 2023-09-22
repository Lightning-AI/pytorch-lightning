from lightning.app.components import TracerPythonScript
from lightning.app.storage.path import Path
from lightning.app.utilities.tracer import Tracer
from lightning.pytorch import Trainer


class PLTracerPythonScript(TracerPythonScript):
    """This component can be used for ANY PyTorch Lightning script to track its progress and extract its best model
    path."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define the component state.
        self.global_step = None
        self.best_model_path = None

    def configure_tracer(self) -> Tracer:
        from lightning.pytorch.callbacks import Callback

        class MyInjectedCallback(Callback):
            def __init__(self, lightning_work):
                self.lightning_work = lightning_work

            def on_train_start(self, trainer, pl_module) -> None:
                print("This code doesn't belong to the script but was injected.")
                print("Even the Lightning Work is available and state transfer works !")
                print(self.lightning_work)

            def on_batch_train_end(self, trainer, *_) -> None:
                # On every batch end, collects some information.
                # This is communicated automatically to the rest of the app,
                # so you can track your training in real time in the Lightning App UI.
                self.lightning_work.global_step = trainer.global_step
                best_model_path = trainer.checkpoint_callback.best_model_path
                if best_model_path:
                    self.lightning_work.best_model_path = Path(best_model_path)

        # This hook would be called every time
        # before a Trainer `__init__` method is called.

        def trainer_pre_fn(trainer, *args, **kwargs):
            kwargs["callbacks"] = kwargs.get("callbacks", []) + [MyInjectedCallback(self)]
            return {}, args, kwargs

        tracer = super().configure_tracer()
        tracer.add_traced(Trainer, "__init__", pre_fn=trainer_pre_fn)
        return tracer


if __name__ == "__main__":
    comp = PLTracerPythonScript(Path(__file__).parent / "pl_script.py")
    res = comp.run()
