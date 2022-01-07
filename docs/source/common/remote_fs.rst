Remote filesystems
==================

PyTorch Lightning enables working with data from a variety of filesystems, including local filesystems and several cloud storage providers
such as ``s3`` on AWS, ``gcs`` on Google Cloud, or ``adl`` on Azure.

This applies to saving and writing checkpoints, as well as for logging.
Working with different filesystems can be accomplished by appending a protocol like "s3:/" to file paths for writing and reading data.


.. code-block:: python

    # `default_root_dir` is the default path used for logs and weights
    trainer = Trainer(default_root_dir="s3://my_bucket/data/")
    trainer.fit(model)

You could pass custom paths to loggers for logging data.

.. code-block:: python

    from pytorch_lightning.loggers import TensorBoardLogger

    logger = TensorBoardLogger(save_dir="s3://my_bucket/logs/")

    trainer = Trainer(logger=logger)
    trainer.fit(model)

Additionally, you could also resume training with a checkpoint stored at a remote filesystem.

.. code-block:: python

    trainer = Trainer(default_root_dir=tmpdir, max_steps=3)
    trainer.fit(model, ckpt_path="s3://my_bucket/ckpts/classifier.ckpt")

PyTorch Lightning uses `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`__ internally to handle all filesystem operations.

The most common filesystems supported by Lightning are:

* Local filesystem: ``file://`` - It's the default and doesn't need any protocol to be used. It's installed by default in Lightning.
* Amazon S3: ``s3://`` - Amazon S3 remote binary store, using the library `s3fs <https://s3fs.readthedocs.io/>`__. Run ``pip install fsspec[s3]`` to install it.
* Google Cloud Storage: ``gcs://`` or ``gs://`` - Google Cloud Storage, using `gcsfs <https://gcsfs.readthedocs.io/en/stable/>`__. Run ``pip install fsspec[gcs]`` to install it.
* Microsoft Azure Storage: ``adl://``, ``abfs://`` or ``az://`` - Microsoft Azure Storage, using `adlfs <https://github.com/fsspec/adlfs>`__. Run ``pip install fsspec[adl]`` to install it.
* Hadoop File System: ``hdfs://`` - Hadoop Distributed File System. This uses `PyArrow <https://arrow.apache.org/docs/python/>`__ as the backend. Run ``pip install fsspec[hdfs]`` to install it.

You could learn more about the available filesystems with:

.. code-block:: python

    from fsspec.registry import known_implementations

    print(known_implementations)


You could also look into :ref:`CheckpointIO Plugin <common/checkpointing:Customize Checkpointing>` for more details on how to customize saving and loading checkpoints.
