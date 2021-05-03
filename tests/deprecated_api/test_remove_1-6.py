import pytest

from pytorch_lightning import Trainer
from tests.helpers import BoringModel


def test_v1_6_0_dataloader_renaming(tmpdir):
    model = BoringModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    dl = model.train_dataloader()

    with pytest.deprecated_call(match=r"fit\(train_dataloader\)` is deprecated in v1.4"):
        trainer.fit(model, train_dataloader=dl)

    with pytest.deprecated_call(match=r"validate\(val_dataloaders\)` is deprecated in v1.4"):
        trainer.validate(model, val_dataloaders=dl)

    with pytest.deprecated_call(match=r"test\(test_dataloaders\)` is deprecated in v1.4"):
        trainer.test(model, test_dataloaders=dl)

    with pytest.deprecated_call(match=r"tune\(train_dataloader\)` is deprecated in v1.4"):
        trainer.tune(model, train_dataloader=dl)
    with pytest.deprecated_call(match=r"tune\(train_dataloader\)` is deprecated in v1.4"):
        trainer.tuner.scale_batch_size(model, train_dataloader=dl)
    with pytest.deprecated_call(match=r"tune\(train_dataloader\)` is deprecated in v1.4"):
        trainer.tuner.lr_find(model, train_dataloader=dl)
