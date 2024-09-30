.. _checkpointing_advanced:

##################################
Cloud-based checkpoints (advanced)
##################################


*****************
Cloud checkpoints
*****************
Lightning is integrated with the major remote file systems including local filesystems and several cloud storage providers such as
`S3 <https://aws.amazon.com/s3/>`_ on `AWS <https://aws.amazon.com/>`_, `GCS <https://cloud.google.com/storage>`_ on `Google Cloud <https://cloud.google.com/>`_,
or `ADL <https://azure.microsoft.com/solutions/data-lake/>`_ on `Azure <https://azure.microsoft.com/>`_.

PyTorch Lightning uses `fsspec <https://filesystem-spec.readthedocs.io/>`_ internally to handle all filesystem operations.

----

Save a cloud checkpoint
=======================

To save to a remote filesystem, prepend a protocol like "s3:/" to the root_dir used for writing and reading model data.

.. code-block:: python

    # `default_root_dir` is the default path used for logs and checkpoints
    trainer = Trainer(default_root_dir="s3://my_bucket/data/")
    trainer.fit(model)

----

Resume training from a cloud checkpoint
=======================================
To resume training from a cloud checkpoint use a cloud url.

.. code-block:: python

    trainer = Trainer(default_root_dir=tmpdir, max_steps=3)
    trainer.fit(model, ckpt_path="s3://my_bucket/ckpts/classifier.ckpt")

PyTorch Lightning uses `fsspec <https://filesystem-spec.readthedocs.io/>`_ internally to handle all filesystem operations.
