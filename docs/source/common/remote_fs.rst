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

The most common filesystems supported by Lightning are:

* Local filesystem: ``file://`` - It's the default and doesn't need any protocol to be used.
* Amazon S3: ``s3://`` - Amazon S3 remote binary store, using the library `s3fs <https://s3fs.readthedocs.io/>`__.
* Google Cloud Storage: ``gcs://`` or ``gs://`` - Google Cloud Storage, using `gcsfs <https://gcsfs.readthedocs.io/en/stable/>`__.
* Microsoft Azure Storage: ``adl://``, ``abfs://`` or ``az://`` - Microsoft Azure Storage, using `adlfs <https://github.com/fsspec/adlfs>`__.
* Hadoop File System: ``hdfs://`` - Hadoop Distributed File System. This uses `PyArrow <https://arrow.apache.org/docs/python/>`__ as the backend.

You could learn more about the available filesystems with:

.. code-block:: python

    from fsspec.registry import known_implementations

    print(known_implementations)
