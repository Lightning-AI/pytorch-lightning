from pytorch_lightning.strategies.executors.base import Executor


class SingleProcessExecutor(Executor):
    def execute(self, function, *args, **kwargs):
        return function(*args, **kwargs)
