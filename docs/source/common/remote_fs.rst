Remote filesystems
==================

PyTorch Lightning enables working with data from a variety of filesystems, including local filesystems and several cloud storage providers.

The datastores could be used to load and save checkpoints, as well as for logging.
Working with different filesystems could be accomplished by appending a protocol like "s:/" to filepaths for writing and reading data.


.. code-block:: python

    # `default_root_dir` is the default path used for logs and weights
    trainer = Trainer(default_root_dir="s3://lightning/data/")
    trainer.fit(model)

You could pass custom paths to loggers for logging data.

.. code-block:: python

    from pytorch_lightning.loggers import TensorBoardLogger

    logger = TensorBoardLogger(save_dir="s3://lightning/logs/")

    trainer = Trainer(logger=logger)
    trainer.fit(model)

Additionally, you could also resume training with a checkpoint stored at a remote filesystem.

.. code-block:: python

    trainer = Trainer(default_root_dir=tmpdir, max_steps=3)
    trainer.fit(model, ckpt_path="s3://lightning/ckpts/classifier.ckpt")

PyTorch Lightning uses `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`__ internally to handle all filesystem operations.
