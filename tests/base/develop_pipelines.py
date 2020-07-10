import torch

from pytorch_lightning import Trainer
from tests.base.develop_utils import load_model_from_checkpoint, get_default_logger, \
    reset_seed


def run_model_test_without_loggers(trainer_options, model, min_acc: float = 0.50):
    reset_seed()

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, 'amp + ddp model failed to complete'

    pretrained_model = load_model_from_checkpoint(
        trainer.logger,
        trainer.checkpoint_callback.best_model_path,
    )

    # test new model accuracy
    test_loaders = model.test_dataloader()
    if not isinstance(test_loaders, list):
        test_loaders = [test_loaders]

    for dataloader in test_loaders:
        run_prediction(dataloader, pretrained_model, min_acc=min_acc)

    if trainer.use_ddp:
        # on hpc this would work fine... but need to hack it for the purpose of the test
        trainer.model = pretrained_model
        trainer.optimizers, trainer.lr_schedulers = pretrained_model.configure_optimizers()


def run_model_test(trainer_options, model, on_gpu: bool = True, version=None, with_hpc: bool = True):

    reset_seed()
    save_dir = trainer_options['default_root_dir']

    # logger file to get meta
    logger = get_default_logger(save_dir, version=version)
    trainer_options.update(logger=logger)

    if 'checkpoint_callback' not in trainer_options:
        trainer_options.update(checkpoint_callback=True)

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, 'trainer failed'

    # test model loading
    pretrained_model = load_model_from_checkpoint(logger, trainer.checkpoint_callback.best_model_path)

    # test new model accuracy
    test_loaders = model.test_dataloader()
    if not isinstance(test_loaders, list):
        test_loaders = [test_loaders]

    for dataloader in test_loaders:
        run_prediction(dataloader, pretrained_model)

    if with_hpc:
        if trainer.use_ddp or trainer.use_ddp2:
            # on hpc this would work fine... but need to hack it for the purpose of the test
            trainer.model = pretrained_model
            trainer.optimizers, trainer.lr_schedulers, trainer.optimizer_frequencies = \
                trainer.init_optimizers(pretrained_model)

        # test HPC loading / saving
        trainer.hpc_save(save_dir, logger)
        trainer.hpc_load(save_dir, on_gpu=on_gpu)


def run_prediction(dataloader, trained_model, dp=False, min_acc=0.50):
    # run prediction on 1 batch
    batch = next(iter(dataloader))
    x, y = batch
    x = x.view(x.size(0), -1)

    if dp:
        with torch.no_grad():
            output = trained_model(batch, 0)
        acc = output['val_acc']
        acc = torch.mean(acc).item()

    else:
        with torch.no_grad():
            y_hat = trained_model(x)
        y_hat = y_hat.cpu()

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)

        y = y.cpu()
        acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        acc = torch.tensor(acc)
        acc = acc.item()

    assert acc >= min_acc, f"This model is expected to get > {min_acc} in test set (it got {acc})"
