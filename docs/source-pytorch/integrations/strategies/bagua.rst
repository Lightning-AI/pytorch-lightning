:orphan:

#####
Bagua
#####

The `Bagua strategy <https://github.com/Lightning-AI/lightning-Bagua>`_ speeds up PyTorch training from a single node to large scale.
Bagua is a deep learning training acceleration framework for PyTorch, with advanced distributed training algorithms and system optimizations.
Bagua currently supports:

- **Advanced Distributed Training Algorithms**: Users can extend the training on a single GPU to multi-GPUs (may across multiple machines) by simply adding a few lines of code (optionally in `elastic mode <https://tutorials.baguasys.com/elastic-training/>`_). One prominent feature of Bagua is to provide a flexible system abstraction that supports state-of-the-art system relaxation techniques of distributed training. So far, Bagua has integrated communication primitives including

  - Centralized Synchronous Communication (e.g. `Gradient AllReduce <https://tutorials.baguasys.com/algorithms/gradient-allreduce>`_)

  - Decentralized Synchronous Communication (e.g. `Decentralized SGD <https://tutorials.baguasys.com/algorithms/decentralized>`_)

  - Low Precision Communication (e.g. `ByteGrad <https://tutorials.baguasys.com/algorithms/bytegrad>`_)

  - Asynchronous Communication (e.g. `Async Model Average <https://tutorials.baguasys.com/algorithms/async-model-average>`_)
- `Cached Dataset <https://tutorials.baguasys.com/more-optimizations/cached-dataset>`_: When samples in a dataset need tedious preprocessing, or reading the dataset itself is slow, they could become a major bottleneck of the whole training process. Bagua provides cached dataset to speedup this process by caching data samples in memory, so that reading these samples after the first time can be much faster.
- `TCP Communication Acceleration (Bagua-Net) <https://tutorials.baguasys.com/more-optimizations/bagua-net>`_: Bagua-Net is a low level communication acceleration feature provided by Bagua. It can greatly improve the throughput of AllReduce on TCP network. You can enable Bagua-Net optimization on any distributed training job that uses NCCL to do GPU communication (this includes PyTorch-DDP, Horovod, DeepSpeed, and more).
- `Performance Autotuning <https://tutorials.baguasys.com/performance-autotuning/>`_: Bagua can automatically tune system parameters to achieve the highest throughput.
- `Generic Fused Optimizer <https://tutorials.baguasys.com/more-optimizations/generic-fused-optimizer>`_: Bagua provides generic fused optimizer which improves the performance of optimizers, by fusing the optimizer `.step()` operation on multiple layers. It can be applied to arbitrary PyTorch optimizer, in contrast to `NVIDIA Apex <https://nvidia.github.io/apex/optimizers.html>`_'s approach, where only some specific optimizers are implemented.
- `Load Balanced Data Loader <https://tutorials.baguasys.com/more-optimizations/load-balanced-data-loader>`_: When the computation complexity of samples in training data are different, for example in NLP and speech tasks, where each sample have different lengths, distributed training throughput can be greatly improved by using Bagua's load balanced data loader, which distributes samples in a way that each worker's workload are similar.

You can install the Bagua integration by running

.. code-block:: bash

    pip install lightning-bagua

This will install both the `bagua <https://pypi.org/project/bagua/>`_ package as well as the ``BaguaStrategy`` for the Lightning Trainer:

.. code-block:: python

    trainer = Trainer(strategy="bagua", accelerator="gpu", devices=...)


You can tune several settings by instantiating the strategy objects and pass options in:

.. code-block:: python

    from lightning_bagua import BaguaStrategy

    strategy = BaguaStrategy(algorithm="bytegrad")
    trainer = Trainer(strategy=strategy, accelerator="gpu", devices=...)


.. note::

    *  Bagua is only supported on Linux systems with GPU(s).

See `Bagua Tutorials <https://tutorials.baguasys.com/>`_ for more details on installation and advanced features.
