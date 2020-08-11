import pytest
import torch
from packaging.version import parse
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import IterableDataset

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.data import has_iterable_dataset, has_len
from tests.base import EvalModelTemplate


@pytest.mark.xfail(
    parse(torch.__version__) < parse("1.4.0"),
    reason="IterableDataset with __len__ before 1.4 raises",
)
def test_warning_with_iterable_dataset_and_len(tmpdir):
    """ Tests that a warning messages is shown when an IterableDataset defines `__len__`. """
    model = EvalModelTemplate()
    original_dataset = model.train_dataloader().dataset

    class IterableWithLen(IterableDataset):

        def __iter__(self):
            return iter(original_dataset)

        def __len__(self):
            return len(original_dataset)

    dataloader = DataLoader(IterableWithLen(), batch_size=16)
    assert has_len(dataloader)
    assert has_iterable_dataset(dataloader)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_steps=3,
    )
    with pytest.warns(UserWarning, match='Your `IterableDataset` has `__len__` defined.'):
        trainer.fit(model, train_dataloader=dataloader, val_dataloaders=[dataloader])
    with pytest.warns(UserWarning, match='Your `IterableDataset` has `__len__` defined.'):
        trainer.test(model, test_dataloaders=[dataloader])
