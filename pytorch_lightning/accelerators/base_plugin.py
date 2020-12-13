import contextlib
import torch

class Plugin(object):

    def connect(self, model: torch.nn.Module, *args, **kwargs):
        pass

    def pre_optimizer_step(self, optimizer, optimizer_idx):
        pass

    def post_optimizer_step(self, optimizer, optimizer_idx):
        pass

    def pre_training(self):
        pass

    def post_training(self, best_model_path):
        pass

    @contextlib.contextmanager
    def train_step_context(self):
        yield

    @contextlib.contextmanager
    def val_step_context(self):
        yield

    @contextlib.contextmanager
    def test_step_context(self):
        yield