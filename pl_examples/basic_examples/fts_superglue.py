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
"""A demonstration of the scheduled finetuning callback
:ref:`FinetuningScheduler<advanced/finetuning_scheduler:Finetuning Scheduler>` using the
`RTE <https://huggingface.co/datasets/viewer/?dataset=super_glue&config=rte>`_ and
`BoolQ <https://github.com/google-research-datasets/boolean-questions>`_ tasks of the
`SuperGLUE <https://super.gluebenchmark.com/>`_ benchmark and the :ref:`LightningCLI<common/lightning_cli:LightningCLI>`

There are three different demo schedule configurations composed with shared defaults (./config/fts/fts_defaults.yaml)
provided for the default 'rte' task. Note DDP w/ 2 GPUs is the default configuration so ensure you adjust the
configuration files referenced below as desired for other configurations.

.. code-block:: bash

    # Generate a baseline without scheduled finetuning enabled:
    python fts_superglue.py fit --config config/fts/nofts_baseline.yaml

    # Train with the default finetuning schedule:
    python fts_superglue.py fit --config config/fts/fts_implicit.yaml

    # Train with a non-default finetuning schedule:
    python fts_superglue.py fit --config config/fts/fts_explicit.yaml
"""

import os
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pl_examples import _HF_AVAILABLE
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.cli import instantiate_class, LightningCLI

if _HF_AVAILABLE:
    import datasets
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

TASK_NUM_LABELS = {"boolq": 2, "rte": 2}
DEFAULT_TASK = "rte"


class RteBoolqModule(pl.LightningModule):
    """A :class:`~pytorch_lightning.core.lightning.LightningModule` that can be used to finetune a foundational
    model on either the RTE or BoolQ `SuperGLUE <https://super.gluebenchmark.com/>`_ tasks using Hugging Face
    implementations of a given model and the `SuperGLUE Hugging Face dataset.

    <https://huggingface.co/datasets/super_glue#data-instances>`_.
    """

    def __init__(
        self,
        model_name_or_path: str,
        optimizer_init: Dict[str, Any],
        lr_scheduler_init: Dict[str, Any],
        pl_lrs_cfg: Optional[Dict[str, Any]] = None,
        model_cfg: Optional[Dict[str, Any]] = None,
        task_name: str = DEFAULT_TASK,
        experiment_tag: str = "default",
    ):
        """In this example, this :class:`~pytorch_lightning.core.lightning.LightningModule` is initialized by composing
        the ./config/fts/fts_defaults.yaml default configuration with various scheduled finetuning yaml configurations
        via the :class:`~pytorch_lightning.utilities.cli.LightningCLI` but it can be used like any other
        :class:`~pytorch_lightning.core.lightning.LightningModule` as well.

        Args:
            model_name_or_path (str): Path to pretrained model or identifier `from <https://huggingface.co/models>`_
            optimizer_init (Dict[str, Any]): The desired optimizer configuration.
            lr_scheduler_init (Dict[str, Any]): The desired learning rate scheduler config
            pl_lrs_cfg (Optional[Dict[str, Any]]): Defines custom overrides of pytorch lightning lr_scheduler defaults
                defined in :func:`~pytorch_lighting.optimizers._get_default_scheduler_config`
                Example::

                .. code-block:: yaml

                pl_lrs_cfg:
                    interval: epoch
                    frequency: 1
                    name: CosineAnnealingWithWarmRestartsLR

            model_cfg (Optional[Dict[str, Any]], optional): Defines overrides of the default model config. Defaults to
                ``None``.
            task_name (str, optional): The SuperGLUE task to execute, one of ``'rte'``, ``'boolq'``. Defaults to "rte".
            experiment_tag (str, optional): The tag to use for the experiment and tensorboard logs. Defaults to
                "default".
        """
        super().__init__()
        self.optimizer_init = optimizer_init
        self.lr_scheduler_init = lr_scheduler_init
        self.pl_lrs_cfg = pl_lrs_cfg or {}
        if task_name in TASK_NUM_LABELS.keys():
            self.task_name = task_name
        else:
            self.task_name = DEFAULT_TASK
            rank_zero_warn(f"Invalid task_name '{task_name}'. Proceeding with the default task: '{DEFAULT_TASK}'")
        self.num_labels = TASK_NUM_LABELS[self.task_name]
        self.save_hyperparameters()
        self.experiment_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{experiment_tag}"
        self.model_cfg = model_cfg or {}
        conf = AutoConfig.from_pretrained(model_name_or_path, num_labels=self.num_labels, local_files_only=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=conf)
        self.model.config.update(self.model_cfg)  # apply model config overrides
        self.metric = datasets.load_metric("super_glue", self.task_name, experiment_id=self.experiment_id)
        self.no_decay = ["bias", "LayerNorm.weight"]

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        if self.trainer.finetuning_scheduler_callback:
            self.log("finetuning_schedule_depth", self.trainer.finetuning_scheduler_callback.curr_depth)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]
        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        metric_dict = self.metric.compute(predictions=preds, references=labels)
        self.log_dict(metric_dict, prog_bar=True, sync_dist=True)

    def init_pgs(self) -> List[Dict]:
        """Initialize the parameter groups, necessary only for the baseline 'nofts_baseline.yaml' configuration
        that doesn't use the :class:`~pytorch_lightning.callbacks.FinetuningScheduler` callback.

        Returns:
            List[Dict]: A list of parameter group dictionaries.
        """
        pgs = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in self.no_decay) and p.requires_grad
                ],
                "weight_decay": self.optimizer_init["init_args"]["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in self.no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        return pgs

    def configure_optimizers(self):
        optimizer = instantiate_class(self.init_pgs(), self.optimizer_init)
        scheduler = {"scheduler": instantiate_class(optimizer, self.lr_scheduler_init), **self.pl_lrs_cfg}
        return [optimizer], [scheduler]


class RteBoolqDataModule(pl.LightningDataModule):
    """A :class:`~pytorch_lighting.core.LightningDataModule` for using either the RTE or BoolQ `SuperGLUE Hugging
    Face datasets <https://huggingface.co/datasets/super_glue#data-instances>`_."""

    task_text_field_map = {"rte": ["premise", "hypothesis"], "boolq": ["question", "passage"]}
    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]
    # ignore warnings related tokenizers_parallelism/DataLoader parallelism tradeoff and expected logging behavior
    for warnf in [".*does not have many workers*", ".*The number of training samples.*"]:
        warnings.filterwarnings("ignore", warnf)

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = DEFAULT_TASK,
        prep_on_init: bool = False,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        pin_memory: bool = False,
        tokenizers_parallelism: str = "true",
        num_workers: int = 0,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name if task_name in TASK_NUM_LABELS.keys() else DEFAULT_TASK
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizers_parallelism = tokenizers_parallelism
        self.dataloader_kwargs = {"num_workers": num_workers, "pin_memory": pin_memory}
        self.text_fields = self.task_text_field_map[self.task_name]
        self.num_labels = TASK_NUM_LABELS[self.task_name]
        os.environ["TOKENIZERS_PARALLELISM"] = self.tokenizers_parallelism
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True, local_files_only=False)
        if prep_on_init:  # useful if one wants to load datasets as soon as the `LightningDataModule` is instantiated
            self.prepare_data()
            self.setup("fit")

    def setup(self, stage):
        self.dataset = datasets.load_dataset("super_glue", self.task_name)
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features, batched=True, remove_columns=["label"]
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        # N.B. PL calls prepare_data from a single process (rank 0) so do not use it to assign state (e.g. self.x=y)
        datasets.load_dataset("super_glue", self.task_name)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, **self.dataloader_kwargs)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size, **self.dataloader_kwargs)
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(self.dataset[x], batch_size=self.eval_batch_size, **self.dataloader_kwargs)
                for x in self.eval_splits
            ]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, **self.dataloader_kwargs)
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(self.dataset[x], batch_size=self.eval_batch_size, **self.dataloader_kwargs)
                for x in self.eval_splits
            ]

    def convert_to_features(self, example_batch):
        text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            text_pairs, max_length=self.max_seq_length, padding="longest", truncation=True
        )
        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]
        return features


class CustLightningCLI(LightningCLI):
    """Customize the :class:`~pytorch_lightning.utilities.cli.LightningCLI` to ensure the
    :class:`~pytorch_lighting.core.LightningDataModule` and :class:`~pytorch_lightning.core.lightning.LightningModule`
    use the same Hugging Face model, SuperGLUE task and custom logging tag."""

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.logger.init_args.name", "model.init_args.experiment_tag")
        parser.link_arguments("data.init_args.model_name_or_path", "model.init_args.model_name_or_path")
        parser.link_arguments("data.init_args.task_name", "model.init_args.task_name")


def cli_main() -> None:
    if not _HF_AVAILABLE:  # pragma: no cover
        print("Running the fts_superglue example requires the `transformers` and `datasets` packages from Hugging Face")
        return
    # every configuration of this example depends upon a shared set of defaults.
    default_config_file = os.path.join(os.path.dirname(__file__), "config", "fts", "fts_defaults.yaml")
    _ = CustLightningCLI(
        pl.LightningModule,
        pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_overwrite=True,
        parser_kwargs={"fit": {"default_config_files": [default_config_file]}},
    )


if __name__ == "__main__":
    cli_main()
