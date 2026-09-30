.. _remote_fs:

##################
Remote Filesystems
##################

PyTorch Lightning enables working with data from a variety of filesystems, including local filesystems and several cloud storage providers such as
`S3 <https://aws.amazon.com/s3/>`_ on `AWS <https://aws.amazon.com/>`_, `GCS <https://cloud.google.com/storage>`_ on `Google Cloud <https://cloud.google.com/>`_,
or `ADL <https://azure.microsoft.com/solutions/data-lake/>`_ on `Azure <https://azure.microsoft.com/>`_.

This applies to saving and writing checkpoints, as well as for logging.
Working with different filesystems can be accomplished by appending a protocol like "s3:/" to file paths for writing and reading data.

.. code-block:: python

    # `default_root_dir` is the default path used for logs and checkpoints
    trainer = Trainer(default_root_dir="s3://my_bucket/data/")
    trainer.fit(model)


For logging, remote filesystem support depends on the particular logger integration being used. Consult :ref:`the documentation of the individual logger <loggers-api-references>` for more details.

.. code-block:: python

    from lightning.pytorch.loggers import TensorBoardLogger

    logger = TensorBoardLogger(save_dir="s3://my_bucket/logs/")

    trainer = Trainer(logger=logger)
    trainer.fit(model)

Additionally, you could also resume training with a checkpoint stored at a remote filesystem.

.. code-block:: python

    trainer = Trainer(default_root_dir=tmpdir, max_steps=3)
    trainer.fit(model, ckpt_path="s3://my_bucket/ckpts/classifier.ckpt")

.. note::
    When loading a remote checkpoint of 128 MB or larger, Lightning downloads it once into a
    node-local cache directory (preferring the ``/dev/shm`` RAM disk when it has enough free
    capacity, otherwise the system temporary directory) and then loads it with memory-mapping
    (``mmap=True``). Ranks sharing a node elect a single downloader through a POSIX file lock, so
    the object is fetched once and the resulting pages are shared by all of them. Platforms without
    ``fcntl`` (Windows) stream the checkpoint instead.

    Cache entries are keyed on the remote object's version (``etag``, ``generation`` or
    ``version_id``), so overwriting a checkpoint at the same path invalidates the old entry
    automatically and the superseded copy is reclaimed. If the backend reports no version
    information, the checkpoint is streamed instead of cached.

    Entries outlive the process so that later jobs on the same node reuse them. Nothing is ever
    evicted to make room: a checkpoint is only cached when the root has room for it, so a filling
    root simply stops accepting new entries and later loads stream instead. ``/dev/shm`` is cleared
    on reboot, and ``lightning.fabric.utilities.cloud_io.clear_cache()`` reclaims the cached
    checkpoints on demand (empty lock marker files are left behind, since removing one is unsafe
    while another rank may be waiting on it). An entry in ``/dev/shm`` stays resident in RAM and,
    in containers, counts toward the container's memory limit even after the process exits; point
    the cache at local disk or disable it if that headroom is tight. Two environment variables
    control the behavior:

    .. list-table::
        :widths: 40 60
        :header-rows: 1

        * - Variable
          - Effect
        * - ``LIGHTNING_CHECKPOINT_CACHE``
          - Set to ``0`` to disable caching and always stream.
        * - ``LIGHTNING_CHECKPOINT_CACHE_DIR``
          - Use this directory instead of ``/dev/shm`` and the temporary directory.

PyTorch Lightning uses `fsspec <https://filesystem-spec.readthedocs.io/>`_ internally to handle all filesystem operations.

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


You could also look into :ref:`CheckpointIO Plugin <checkpointing_expert>` for more details on how to customize saving and loading checkpoints.
