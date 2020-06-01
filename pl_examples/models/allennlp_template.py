import itertools
import statistics
from argparse import Namespace
from copy import deepcopy
from typing import Dict, Union

import allennlp.training.util as training_util
import flatten_dict
import torch
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataloader import DataLoader
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model

from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule


def _is_scalar(x):
    return isinstance(x, (int, float)) or (isinstance(x, torch.Tensor) and x.dim() == 0)


def _is_loggable(x):
    return _is_scalar(x) or isinstance(x, dict) or isinstance(x, list)


def anlp_param(param):
    # Not sure if this is the best way to define the paramaters
    for p in ["train_data_path", "validation_data_path", "test_data_path", "dataset_reader",
              "val_dataset_reader", "model", "data_loader"]:
        if param == p or param.startswith(p + "."):
            return True
    return False


class AllenNlpLightningModule(LightningModule):

    def __init__(self,
                 params: Union[Params, Dict, Namespace] = None,
                 filter_params=True,
                 **kwargs
                 ):
        """
        params -- The preferred way to pass params is via the kwargs. This allows loggers such as WandbLogger
        to use the real parameter name instead of prefixing it with "params." However you may also pass in
        an explicit object either as a Params, Dict, or Namespace. Flat dictionaries will be unflattened
        assuming dot-separated paths

        filter_params -- If True, attempt to filter the parameters to just those used by the AllenNlp model

        kwargs -- The parameters to use to initialize the model

        """
        super().__init__()
        if params is None:
            params = kwargs
        # Convert params to dict
        if isinstance(params, Params):
            params = params.as_flat_dict()
        if isinstance(params, Namespace):
            params = params.__dict__

        # Restore params we are interested in to a Params object
        assert isinstance(params, dict)
        if filter_params:
            params = {k: v for k, v in params.items() if anlp_param(k)}
        # allow allennlp params to be specified as a nested dict or a dot-separated flat dict
        params = Params(flatten_dict.unflatten(params, splitter='dot'))

        # Initialize the DatasetReader and Datasets from params
        log.info("Reading datasets...")
        dataset_reader = DatasetReader.from_params(params['dataset_reader'])
        datasets = training_util.read_all_datasets(
            train_data_path=params['train_data_path'],
            dataset_reader=dataset_reader,
            validation_dataset_reader=dataset_reader,
            validation_data_path=params['validation_data_path'],
            test_data_path=params.get('test_data_path'),
        )

        # Select instances in data for building the Vocabulary
        datasets_for_vocab_creation = params.get('datasets_for_vocab_creation')
        if datasets_for_vocab_creation:
            for key in datasets_for_vocab_creation:
                if key not in datasets:
                    raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {key}")

        instance_generator = (
            instance
            for key, dataset in datasets.items()
            if not datasets_for_vocab_creation or key in datasets_for_vocab_creation
            for instance in dataset
        )

        # Build vocabulary
        log.info("Building vocabulary...")
        vocabulary = Vocabulary.from_params(
            params=params.get('vocabulary', Params({})),
            instances=instance_generator
        )

        self.model = Model.from_params(params=params['model'], vocab=vocabulary)

        # Use the vocabulary to convert strings in the data into their vocabulary ids
        log.info("Indexing data...")
        for dataset in datasets.values():
            dataset.index_with(self.model.vocab)

        # Create the train DataLoader
        self.data_loader = DataLoader.from_params(
            params=deepcopy(params['data_loader']),
            dataset=datasets["train"]
        )

        # Optionally, create the validation DataLoader
        validation_data = datasets.get("validation")
        if validation_data is not None:
            self.validation_data_loader = DataLoader.from_params(
                params=deepcopy(params.get('validation_data_loader', params['data_loader'])),
                dataset=validation_data
            )
        else:
            self.validation_data_loader = None

        # Optionally, create the test DataLoader
        test_data = datasets.get("test")
        if test_data is not None:
            self.test_data_loader = DataLoader.from_params(
                params=deepcopy(params.get('validation_data_loader', params['data_loader'])),
                dataset=test_data
            )
        else:
            self.test_data_loader = None

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def _step(self, batch, batch_idx):
        # AllenNlp models compute loss during their forward method,
        # so we do not have to calculate loss separately here
        output_dict = self(**batch)
        output_dict = self.model.make_output_human_readable(output_dict)

        if 'loss' not in output_dict:
            raise ValueError("AllenNlp model returned dictionary missing the 'loss' key")

        # AllenNlp models are responsible for their own metrics
        output_dict.update(self.model.get_metrics())

        update = {}
        update['log'] = {k: v for k, v in output_dict.items() if _is_loggable(v)}
        update['log'].update(loss=output_dict['loss'], epoch=self.current_epoch)
        #         update['log']['batch'] = batch_idx
        update['progress_bar'] = {k: v for k, v in output_dict.items()
                                  if _is_scalar(v) and k != 'loss'}
        update['progress_bar'].update(batch=batch_idx)
        output_dict.update(update)
        return output_dict

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        result = {'log': {}, 'progress_bar': {}}

        # Handle loss:
        val_loss = torch.stack([x['loss'] for x in outputs]).mean()
        result = {
            'val_loss': val_loss,
            'log': {'val_loss': val_loss},
            'progress_bar': {'val_loss': val_loss}
        }

        # Handle everything else
        for key in outputs[0].keys():
            if key in ['log', 'progress_bar']:
                continue
            val_key = "val_" + key
            if _is_scalar(outputs[0][key]):
                # take mean of scalar values
                if isinstance(outputs[0][key], torch.Tensor):
                    val = torch.stack([x[key] for x in outputs]).mean()
                else:
                    val = statistics.mean([x[key] for x in outputs])
                result[val_key] = val
                result['log'][val_key] = val
                result['progress_bar'][val_key] = val
            elif _is_loggable(outputs[0][key]):
                # chain non-scalar values into a list
                val = list(itertools.chain(*[output[key] for output in outputs]))
                result['log'][val_key] = val

        # Add tracking variables
        result['log']['epoch'] = self.current_epoch
        return result

    def train_dataloader(self):
        return self.data_loader

    def val_dataloader(self):
        return self.validation_data_loader

    # Not used for now, since we initialize training parameters using Pytorch-Lightning
    def configure_optimizers(self):
        from allennlp.training.optimizers import Optimizer
        parameters = [[n, p] for n, p in self.model.named_parameters() if p.requires_grad]
        return Optimizer.from_params(
            params=params['trainer']['optimizer'],
            model_parameters=parameters
        )