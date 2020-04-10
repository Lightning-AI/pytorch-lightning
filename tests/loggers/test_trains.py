import pickle

import tests.base.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TrainsLogger
from tests.base import LightningTestModel


def test_trains_logger(tmpdir):
    """Verify that basic functionality of TRAINS logger works."""
    tutils.reset_seed()

    hparams = tutils.get_default_hparams()
    model = LightningTestModel(hparams)
    TrainsLogger.set_bypass_mode(True)
    TrainsLogger.set_credentials(api_host='http://integration.trains.allegro.ai:8008',
                                 files_host='http://integration.trains.allegro.ai:8081',
                                 web_host='http://integration.trains.allegro.ai:8080', )
    logger = TrainsLogger(project_name="lightning_log", task_name="pytorch lightning test")

    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        train_percent_check=0.05,
        logger=logger
    )
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    print('result finished')
    logger.finalize()
    assert result == 1, "Training failed"


def test_trains_pickle(tmpdir):
    """Verify that pickling trainer with TRAINS logger works."""
    tutils.reset_seed()

    # hparams = tutils.get_default_hparams()
    # model = LightningTestModel(hparams)
    TrainsLogger.set_bypass_mode(True)
    TrainsLogger.set_credentials(api_host='http://integration.trains.allegro.ai:8008',
                                 files_host='http://integration.trains.allegro.ai:8081',
                                 web_host='http://integration.trains.allegro.ai:8080', )
    logger = TrainsLogger(project_name="lightning_log", task_name="pytorch lightning test")

    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        logger=logger
    )

    trainer = Trainer(**trainer_options)
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({"acc": 1.0})
    trainer2.logger.finalize()
    logger.finalize()
