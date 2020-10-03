"""
Tests to ensure that the training loop works with a dict (1.0)
"""
from pytorch_lightning import Trainer
from tests.base.deterministic_model import DeterministicModel
import os
import torch


def test__training_step__log(tmpdir):
    """
    Tests that only training_step can be used
    """
    os.environ['PL_DEV_DEBUG'] = '1'

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx

            # -----------
            # default
            # -----------
            self.log('default', acc)

            # -----------
            # logger
            # -----------
            # on_step T on_epoch F
            self.log('l_s', acc, on_step=True, on_epoch=False, prog_bar=False, logger=True)

            # on_step F on_epoch T
            self.log('l_e', acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            # on_step T on_epoch T
            self.log('l_se', acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

            # -----------
            # pbar
            # -----------
            # on_step T on_epoch F
            self.log('p_s', acc, on_step=True, on_epoch=False, prog_bar=True, logger=False)

            # on_step F on_epoch T
            self.log('p_e', acc, on_step=False, on_epoch=True, prog_bar=True, logger=False)

            # on_step T on_epoch T
            self.log('p_se', acc, on_step=True, on_epoch=True, prog_bar=True, logger=False)

            self.training_step_called = True
            return acc

        def backward(self, trainer, loss, optimizer, optimizer_idx):
            loss.backward()

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        row_log_interval=1,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called
    assert not model.training_step_end_called
    assert not model.training_epoch_end_called

    # make sure all the metrics are available for callbacks
    logged_metrics = set(trainer.logged_metrics.keys())
    expected_logged_metrics = {
        'epoch',
        'default',
        'l_e',
        'l_s',
        'l_se',
        'step_l_se',
        'epoch_l_se',
    }
    assert logged_metrics == expected_logged_metrics

    pbar_metrics = set(trainer.progress_bar_metrics.keys())
    expected_pbar_metrics = {
        'p_e',
        'p_s',
        'p_se',
        'step_p_se',
        'epoch_p_se',
    }
    assert pbar_metrics == expected_pbar_metrics

    callback_metrics = set(trainer.callback_metrics.keys())
    callback_metrics.remove('debug_epoch')
    expected_callback_metrics = set()
    expected_callback_metrics = expected_callback_metrics.union(logged_metrics)
    expected_callback_metrics = expected_callback_metrics.union(pbar_metrics)
    expected_callback_metrics.remove('epoch')
    assert callback_metrics == expected_callback_metrics


def test__training_step__epoch_end__log(tmpdir):
    """
    Tests that only training_step can be used
    """
    os.environ['PL_DEV_DEBUG'] = '1'

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            self.training_step_called = True
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.log('a', acc, on_step=True, on_epoch=True)
            self.log_dict({'a1': acc, 'a2': acc})
            return acc

        def training_epoch_end(self, outputs):
            self.training_epoch_end_called = True
            self.log('b1', outputs[0]['loss'])
            self.log('b', outputs[0]['loss'], on_epoch=True, prog_bar=True, logger=True)

        def backward(self, trainer, loss, optimizer, optimizer_idx):
            loss.backward()

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        row_log_interval=1,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called
    assert not model.training_step_end_called
    assert model.training_epoch_end_called

    # make sure all the metrics are available for callbacks
    logged_metrics = set(trainer.logged_metrics.keys())
    expected_logged_metrics = {
        'epoch',
        'a',
        'step_a',
        'epoch_a',
        'b',
        'b1',
        'a1',
        'a2'
    }
    assert logged_metrics == expected_logged_metrics

    pbar_metrics = set(trainer.progress_bar_metrics.keys())
    expected_pbar_metrics = {
        'b',
    }
    assert pbar_metrics == expected_pbar_metrics

    callback_metrics = set(trainer.callback_metrics.keys())
    callback_metrics.remove('debug_epoch')
    expected_callback_metrics = set()
    expected_callback_metrics = expected_callback_metrics.union(logged_metrics)
    expected_callback_metrics = expected_callback_metrics.union(pbar_metrics)
    expected_callback_metrics.remove('epoch')
    assert callback_metrics == expected_callback_metrics


def test__training_step__step_end__epoch_end__log(tmpdir):
    """
    Tests that only training_step can be used
    """
    os.environ['PL_DEV_DEBUG'] = '1'

    class TestModel(DeterministicModel):
        def training_step(self, batch, batch_idx):
            self.training_step_called = True
            acc = self.step(batch, batch_idx)
            acc = acc + batch_idx
            self.log('a', acc, on_step=True, on_epoch=True)
            return acc

        def training_step_end(self, out):
            self.training_step_end_called = True
            self.log('b', out, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return out

        def training_epoch_end(self, outputs):
            self.training_epoch_end_called = True
            self.log('c', outputs[0]['loss'], on_epoch=True, prog_bar=True, logger=True)

        def backward(self, trainer, loss, optimizer, optimizer_idx):
            loss.backward()

    model = TestModel()
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=2,
        row_log_interval=1,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called
    assert model.training_step_end_called
    assert model.training_epoch_end_called

    # make sure all the metrics are available for callbacks
    logged_metrics = set(trainer.logged_metrics.keys())
    expected_logged_metrics = {
        'a',
        'step_a',
        'epoch_a',
        'b',
        'step_b',
        'epoch_b',
        'c',
        'epoch',
    }
    assert logged_metrics == expected_logged_metrics

    pbar_metrics = set(trainer.progress_bar_metrics.keys())
    expected_pbar_metrics = {'b', 'c', 'epoch_b', 'step_b'}
    assert pbar_metrics == expected_pbar_metrics

    callback_metrics = set(trainer.callback_metrics.keys())
    callback_metrics.remove('debug_epoch')
    expected_callback_metrics = set()
    expected_callback_metrics = expected_callback_metrics.union(logged_metrics)
    expected_callback_metrics = expected_callback_metrics.union(pbar_metrics)
    expected_callback_metrics.remove('epoch')
    assert callback_metrics == expected_callback_metrics
