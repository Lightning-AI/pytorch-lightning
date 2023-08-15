.. _model-parallel:

##################################
Train 1 trillion+ parameter models
##################################

When training large models, fitting larger batch sizes, or trying to increase throughput using multi-GPU compute, Lightning provides advanced optimized distributed training strategies to support these cases and offer substantial improvements in memory usage.

Note that some of the extreme memory saving configurations will affect the speed of training. This Speed/Memory trade-off in most cases can be adjusted.

Some of these memory-efficient strategies rely on offloading onto other forms of memory, such as CPU RAM or NVMe. This means you can even see memory benefits on a **single GPU**, using a strategy such as :ref:`deepspeed-zero-stage-3-offload`.

Check out this amazing video explaining model parallelism and how it works behind the scenes:

.. raw:: html

    <iframe width="540" height="300" src="https://www.youtube.com/embed/w_CKzh5C1K4" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


*********************************************
Choosing an Advanced Distributed GPU Strategy
*********************************************

If you would like to stick with PyTorch DDP, see :ref:`ddp-optimizations`.

Unlike :class:`~torch.nn.parallel.DistributedDataParallel` (DDP) where the maximum trainable model size and batch size do not change with respect to the number of GPUs, memory-optimized strategies can accommodate bigger models and larger batches as more GPUs are used. This means as you scale up the number of GPUs, you can reach the number of model parameters you'd like to train.

There are many considerations when choosing a strategy as described below. In addition, check out the visualization of various strategy benchmarks using `minGPT <https://github.com/SeanNaren/minGPT>`__ `here <https://share.streamlit.io/seannaren/mingpt/streamlit/app.py>`__.

Pre-training vs Fine-tuning
===========================

When fine-tuning, we often use a magnitude less data compared to pre-training a model. This is important when choosing a distributed strategy as usually for pre-training, **we are compute-bound**.
This means we cannot sacrifice throughput as much as if we were fine-tuning, because in fine-tuning the data requirement is smaller.

Overall:

* When **fine-tuning** a model, use advanced memory efficient strategies such as :ref:`fully-sharded-training`, :ref:`deepspeed-zero-stage-3` or :ref:`deepspeed-zero-stage-3-offload`, allowing you to fine-tune larger models if you are limited on compute
* When **pre-training** a model, use simpler optimizations such as :ref:`deepspeed-zero-stage-2`, scaling the number of GPUs to reach larger parameter sizes
* For both fine-tuning and pre-training, use :ref:`deepspeed-activation-checkpointing` as the throughput degradation is not significant

For example when using 128 GPUs, you can **pre-train** large 10 to 20 Billion parameter models using :ref:`deepspeed-zero-stage-2` without having to take a performance hit with more advanced optimized multi-gpu strategy.

But for **fine-tuning** a model, you can reach 10 to 20 Billion parameter models using :ref:`deepspeed-zero-stage-3-offload` on a **single GPU**. This does come with a significant throughput hit, which needs to be weighed accordingly.

When Shouldn't I use an Optimized Distributed Strategy?
=======================================================

Sharding techniques help when model sizes are fairly large; roughly 500M+ parameters is where we've seen benefits. However, in the following cases, we recommend sticking to ordinary distributed strategies

* When your model is small (ResNet50 of around 80M Parameters), unless you are using unusually large batch sizes or inputs.
* Due to high distributed communication between devices, if running on a slow network/interconnect, the training might be much slower than expected and then it's up to you to determince the tradeoff here.


Cutting-edge and third-party Strategies
=======================================

Cutting-edge Lightning strategies are being developed by third-parties outside of Lightning.

If you want to try some of the latest and greatest features for model-parallel training, check out the :doc:`Colossal-AI Strategy <../../integrations/strategies/colossalai>` integration.

Another integration is :doc:`Bagua Strategy <../../integrations/strategies/bagua>`, deep learning training acceleration framework for PyTorch, with advanced distributed training algorithms and system optimizations.

For training on unreliable mixed GPUs across the internet check out the :doc:`Hivemind Strategy <../../integrations/strategies/hivemind>` integration.

----


.. include:: ../model_init.rst


----


.. include:: fsdp.rst


----

.. include:: deepspeed.rst