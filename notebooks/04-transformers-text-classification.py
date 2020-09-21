# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/04-transformers-text-classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="8ag5ANQPJ_j9" colab_type="text"
# # Finetune 🤗 Transformers Models with PyTorch Lightning ⚡
#
# This notebook will use HuggingFace's `datasets` library to get data, which will be wrapped in a `LightningDataModule`. Then, we write a class to perform text classification on any dataset from the[ GLUE Benchmark](https://gluebenchmark.com/). (We just show CoLA and MRPC due to constraint on compute/disk)
#
# [HuggingFace's NLP Viewer](https://huggingface.co/nlp/viewer/?dataset=glue&config=cola) can help you get a feel for the two datasets we will use and what tasks they are solving for.
#
# ---
#   - Give us a ⭐ [on Github](https://www.github.com/PytorchLightning/pytorch-lightning/)
#   - Check out [the documentation](https://pytorch-lightning.readthedocs.io/en/latest/)
#   - Join us [on Slack](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A)
#
#   - [HuggingFace nlp](https://github.com/huggingface/nlp)
#   - [HuggingFace transformers](https://github.com/huggingface/transformers)

# + [markdown] id="fqlsVTj7McZ3" colab_type="text"
# ### Setup

# + id="OIhHrRL-MnKK" colab_type="code" colab={}
# !pip install pytorch-lightning datasets transformers

# + id="6yuQT_ZQMpCg" colab_type="code" colab={}
from argparse import ArgumentParser
from datetime import datetime
from typing import Optional

import nlp
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    glue_compute_metrics
)


# + [markdown] id="9ORJfiuiNZ_N" colab_type="text"
# ## GLUE DataModule

# + id="jW9xQhZxMz1G" colab_type="code" colab={}
class GLUEDataModule(pl.LightningDataModule):

    task_text_field_map = {
        'cola': ['sentence'],
        'sst2': ['sentence'],
        'mrpc': ['sentence1', 'sentence2'],
        'qqp': ['question1', 'question2'],
        'stsb': ['sentence1', 'sentence2'],
        'mnli': ['premise', 'hypothesis'],
        'qnli': ['question', 'sentence'],
        'rte': ['sentence1', 'sentence2'],
        'wnli': ['sentence1', 'sentence2'],
        'ax': ['premise', 'hypothesis']
    }

    glue_task_num_labels = {
        'cola': 2,
        'sst2': 2,
        'mrpc': 2,
        'qqp': 2,
        'stsb': 1,
        'mnli': 3,
        'qnli': 2,
        'rte': 2,
        'wnli': 2,
        'ax': 3
    }

    loader_columns = [
        'nlp_idx',
        'input_ids',
        'token_type_ids',
        'attention_mask',
        'start_positions',
        'end_positions',
        'labels'
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str ='mrpc',
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage):
        self.dataset = nlp.load_dataset('glue', self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=['label'],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if 'validation' in x]

    def prepare_data(self):
        nlp.load_dataset('glue', self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
    
    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.train_batch_size)
    
    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset['validation'], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset['test'], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):

        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs,
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features['labels'] = example_batch['label']

        return features


# + [markdown] id="jQC3a6KuOpX3" colab_type="text"
# #### You could use this datamodule with standalone PyTorch if you wanted...

# + id="JCMH3IAsNffF" colab_type="code" colab={}
dm = GLUEDataModule('distilbert-base-uncased')
dm.prepare_data()
dm.setup('fit')
next(iter(dm.train_dataloader()))


# + [markdown] id="l9fQ_67BO2Lj" colab_type="text"
# ## GLUE Model

# + id="gtn5YGKYO65B" colab_type="code" colab={}
class GLUETransformer(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = nlp.load_metric(
            'glue',
            self.hparams.task_name,
            experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return pl.TrainResult(loss)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {'loss': val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        if self.hparams.task_name == 'mnli':
            for i, output in enumerate(outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split('_')[-1]
                preds = torch.cat([x['preds'] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x['labels'] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x['loss'] for x in output]).mean()
                if i == 0:
                    result = pl.EvalResult(checkpoint_on=loss)
                result.log(f'val_loss_{split}', loss, prog_bar=True)
                split_metrics = {f"{k}_{split}": v for k, v in self.metric.compute(preds, labels).items()}
                result.log_dict(split_metrics, prog_bar=True)
            return result

        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, prog_bar=True)
        result.log_dict(self.metric.compute(preds, labels), prog_bar=True)
        return result

    def setup(self, stage):
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = (
                (len(train_loader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.gpus)))
                // self.hparams.accumulate_grad_batches
                * float(self.hparams.max_epochs)
            )

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        return parser


# + [markdown] id="ha-NdIP_xbd3" colab_type="text"
# ### ⚡ Quick Tip 
#   - Combine arguments from your DataModule, Model, and Trainer into one for easy and robust configuration

# + id="3dEHnl3RPlAR" colab_type="code" colab={}
def parse_args(args=None):
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GLUEDataModule.add_argparse_args(parser)
    parser = GLUETransformer.add_model_specific_args(parser)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args(args)


def main(args):
    pl.seed_everything(args.seed)
    dm = GLUEDataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup('fit')
    model = GLUETransformer(num_labels=dm.num_labels, eval_splits=dm.eval_splits, **vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    return dm, model, trainer


# + [markdown] id="PkuLaeec3sJ-" colab_type="text"
# # Training

# + [markdown] id="QSpueK5UPsN7" colab_type="text"
# ## CoLA
#
# See an interactive view of the CoLA dataset in [NLP Viewer](https://huggingface.co/nlp/viewer/?dataset=glue&config=cola)

# + id="NJnFmtpnPu0Y" colab_type="code" colab={}
mocked_args = """
    --model_name_or_path albert-base-v2
    --task_name cola
    --max_epochs 3
    --gpus 1""".split()

args = parse_args(mocked_args)
dm, model, trainer = main(args)
trainer.fit(model, dm)

# + [markdown] id="_MrNsTnqdz4z" colab_type="text"
# ## MRPC
#
# See an interactive view of the MRPC dataset in [NLP Viewer](https://huggingface.co/nlp/viewer/?dataset=glue&config=mrpc)

# + id="LBwRxg9Cb3d-" colab_type="code" colab={}
mocked_args = """
    --model_name_or_path distilbert-base-cased
    --task_name mrpc
    --max_epochs 3
    --gpus 1""".split()

args = parse_args(mocked_args)
dm, model, trainer = main(args)
trainer.fit(model, dm)

# + [markdown] id="iZhbn0HzfdCu" colab_type="text"
# ## MNLI
#
#  - The MNLI dataset is huge, so we aren't going to bother trying to train it here.
#
#  - Let's just make sure our multi-dataloader logic is right by skipping over training and going straight to validation.
#
# See an interactive view of the MRPC dataset in [NLP Viewer](https://huggingface.co/nlp/viewer/?dataset=glue&config=mnli)

# + id="AvsZMOggfcWW" colab_type="code" colab={}
mocked_args = """
    --model_name_or_path distilbert-base-uncased
    --task_name mnli
    --max_epochs 1
    --gpus 1
    --limit_train_batches 10
    --progress_bar_refresh_rate 20""".split()

args = parse_args(mocked_args)
dm, model, trainer = main(args)
trainer.fit(model, dm)
