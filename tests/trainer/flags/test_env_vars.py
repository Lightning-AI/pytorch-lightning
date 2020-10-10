import os

from pytorch_lightning import Trainer


def test_passing_env_variables(tmpdir):
    """Testing overwriting trainer arguments """
    trainer = Trainer()
    assert trainer.logger is not None
    assert trainer.max_steps is None
    trainer = Trainer(False, max_steps=42)
    assert trainer.logger is None
    assert trainer.max_steps == 42

    os.environ['PL_TRAINER_LOGGER'] = 'False'
    os.environ['PL_TRAINER_MAX_STEPS'] = '7'
    trainer = Trainer()
    assert trainer.logger is None
    assert trainer.max_steps == 7

    os.environ['PL_TRAINER_LOGGER'] = 'True'
    trainer = Trainer(False, max_steps=42)
    assert trainer.logger is not None
    assert trainer.max_steps == 7

    # this has to be cleaned
    del os.environ['PL_TRAINER_LOGGER']
    del os.environ['PL_TRAINER_MAX_STEPS']
