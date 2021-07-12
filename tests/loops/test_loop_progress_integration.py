from pytorch_lightning import Trainer
from tests.loops.test_loops import _collect_loop_progress


def test_loop_progress_integration():
    trainer = Trainer()
    # check no progresses are shared
    assert trainer.validate_loop.progress is not trainer.test_loop.progress
    assert trainer.test_loop.progress is not trainer.predict_loop.progress
    # check the validation progresses are not shared
    assert trainer.fit_loop.epoch_loop.val_loop.progress is not trainer.validate_loop.progress
    expected = _collect_loop_progress(trainer.fit_loop)["epoch_loop"]["progress"]
    assert expected == trainer.fit_loop.epoch_loop.progress
    expected = _collect_loop_progress(trainer.fit_loop)["epoch_loop"]["batch_loop"]["progress"]
    assert expected == trainer.fit_loop.epoch_loop.batch_loop.progress
    expected = _collect_loop_progress(trainer.fit_loop)["epoch_loop"]["val_loop"]["progress"]
    assert expected == trainer.fit_loop.epoch_loop.val_loop.progress
    expected = _collect_loop_progress(trainer.fit_loop)["epoch_loop"]["val_loop"]["epoch_loop"]["progress"]
    assert expected == trainer.fit_loop.epoch_loop.val_loop.epoch_loop.progress
