from pytorch_lightning import Trainer


def test_loop_progress_integration():
    trainer = Trainer()
    fit_loop = trainer.fit_loop
    # check identities inside the fit loop
    assert fit_loop.progress.epoch is fit_loop.epoch_loop.progress
    assert fit_loop.epoch_loop.progress.batch is fit_loop.epoch_loop.batch_loop.progress
    assert fit_loop.epoch_loop.progress.optim is fit_loop.epoch_loop.batch_loop.optim_progress
    assert fit_loop.epoch_loop.progress.val is fit_loop.epoch_loop.val_loop.progress
    assert fit_loop.epoch_loop.val_loop.progress.epoch is fit_loop.epoch_loop.val_loop.epoch_loop.progress
    # check identities inside the evaluation and predict loops
    assert trainer.validate_loop.progress.epoch is trainer.validate_loop.epoch_loop.progress
    assert trainer.test_loop.progress.epoch is trainer.test_loop.epoch_loop.progress
    assert trainer.predict_loop.progress.epoch is trainer.predict_loop.epoch_loop.progress
    # check no progresses are shared
    assert trainer.fit_loop.progress is not trainer.validate_loop.progress
    assert trainer.validate_loop.progress is not trainer.test_loop.progress
    assert trainer.test_loop.progress is not trainer.predict_loop.progress
    # check the validation progresses are not shared
    assert trainer.fit_loop.epoch_loop.val_loop.progress is not trainer.validate_loop.progress
