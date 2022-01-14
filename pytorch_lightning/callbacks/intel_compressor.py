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
r"""
Intel Quantization (TODO: pruning)
^^^^^^^^^^^^

"""
from enum import Enum

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.core import LightningDataModule
from pytorch_lightning.trainer.states import TrainerFn, TrainerStatus
from pytorch_lightning.utilities import _NEURAL_COMPRESSOR_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _NEURAL_COMPRESSOR_AVAILABLE:
    from neural_compressor.conf.config import Quantization_Conf
    from neural_compressor.experimental import Quantization


class QuantizationMode(Enum):
    PTQ_DYNAMIC = "post_training_dynamic_quant"
    PTQ_STATIC = "post_training_static_quant"
    QAT = "quant_aware_training"


class INCQuantization(Callback):
    """Quantization allows speeding up inference and decreasing memory requirements by performing computations and
    storing tensors at lower bitwidths (such as INT8) than floating point precision. And this callback will
    quantized model with Intel Neural Compressor(INC) tool.

    Args:

        config_path_or_obj: config file or Quantization_Conf class for INC.

        monitor: Specified metric which will be computed in evaluation function.

        module_name_to_quant: the model name which you want to quantize, it should be torch.nn.Module.

        dirpath: save path where the quantization config and weights be saved to.

    .. Intel Neural Compressor: https://github.com/intel/neural-compressor
    """

    def __init__(
        self,
        config_path_or_obj,
        monitor: str,
        module_name_to_quant: str,
        datamodule: LightningDataModule = None,
        dirpath: str = None,
    ) -> None:
        if not _NEURAL_COMPRESSOR_AVAILABLE:
            raise ModuleNotFoundError(
                "`INCQuantization` requires `neural-compressor`. Install it by running `pip install neural-compressor`."
            )

        self.config = (
            config_path_or_obj
            if isinstance(config_path_or_obj, Quantization_Conf)
            else Quantization_Conf(config_path_or_obj)
        )
        self.approach = self.config.usr_cfg.quantization.approach
        assert monitor is not None, "Please set metric name for evaluation loop!"
        self.monitor = monitor
        assert (
            module_name_to_quant is not None
        ), "Please set module_name_to_quant with module name which you want to quantize"
        self.module_name_to_quant = module_name_to_quant
        self.datamodule = datamodule
        self.dirpath = dirpath

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the compressor loop begins."""

        def eval_func(model):
            setattr(pl_module, self.module_name_to_quant, model)
            out = trainer.validate(pl_module, self.datamodule.val_dataloader() if self.datamodule is not None else None)
            return out[0][self.monitor]

        trainer.quantizing = True

        if self.dirpath is None:
            self.dirpath = trainer.default_root_dir

        quantizer = Quantization(self.config)
        quantizer.model = getattr(pl_module, self.module_name_to_quant)

        if self.approach == QuantizationMode.PTQ_STATIC.value:
            quantizer.calib_dataloader = (
                pl_module.train_dataloader() if self.datamodule is None else self.datamodule.train_dataloader()
            )
        elif self.approach != QuantizationMode.PTQ_DYNAMIC.value:
            raise MisconfigurationException("Unsupport quantization approach:" + self.approach)

        quantizer.eval_func = eval_func
        model = quantizer()
        setattr(pl_module, self.module_name_to_quant, model.model)
        if self.dirpath is not None:
            model.save(self.dirpath)

        trainer.limit_train_batches = 0
        trainer.limit_val_batches = 0
        trainer.state.fn = TrainerFn.FITTING
        trainer.state.status = TrainerStatus.RUNNING
        trainer.training = True
