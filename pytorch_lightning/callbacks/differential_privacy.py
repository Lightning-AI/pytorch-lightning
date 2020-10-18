# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Differential Privacy
====================

Train your model with differential privacy using Opacus (https://github.com/pytorch/opacus).

"""
import importlib

from pytorch_lightning.callbacks.base import Callback

OPACUS_INSTALLED = importlib.util.find_spec("opacus")

if OPACUS_INSTALLED:
    from opacus import PrivacyEngine
    from opacus.dp_model_inspector import DPModelInspector
    from opacus.utils import module_modification


class DifferentialPrivacy(Callback):
    r"""
    Attach privacy engine to the optimizer before the training begins.
    Args:
        alphas : List[float]
            A list of RDP orders
        noise_multiplier : float
            The ratio of the standard deviation of the Gaussian noise to
            the L2-sensitivity of the function to which the noise is added
        max_grad_norm : Union[float, List[float]]
            The maximum norm of the per-sample gradients. Any gradient with norm
            higher than this will be clipped to this value.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import DifferentialPrivacy
        >>> differential_privacy = DifferentialPrivacy(noise_multiplier=0.3, max_grad_norm=0.1)
        >>> trainer = Trainer(callbacks=[differential_privacy])
    """

    def __init__(
        self,
        alphas=tuple(1 + x / 10.0 for x in range(1, 100)) + tuple(range(12, 64)),
        noise_multiplier=0.1,
        max_grad_norm=0.1,
        **kwargs,
    ):
        if not OPACUS_INSTALLED:
            raise ImportError(
                "This callback requires `opacus` which is not installed yet,"
                " install it with `pip install opacus`."
            )
        super().__init__()
        self.alphas = alphas
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.kwargs = kwargs

    def on_train_start(self, trainer, pl_module):
        """
        Check model compatibility and attach privacy engine to optimizer before training
        """

        # TODO: explore VIRTUAL_BATCH_SIZE
        # TODO: tune max grad
        # TODO: check that channels divisible by 32
        # TODO: get_privacy_spent

        trainer.model = module_modification.convert_batchnorm_modules(pl_module)
        inspector = DPModelInspector()
        inspector.validate(trainer.model)

        privacy_engine = PrivacyEngine(
            pl_module,
            batch_size=trainer.train_dataloader.batch_size,
            sample_size=len(trainer.train_dataloader.dataset),
            alphas=self.alphas,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            **self.kwargs,
        )

        for optimizer in pl_module.optimizers():
            privacy_engine.attach(optimizer)
