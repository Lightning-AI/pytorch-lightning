.. _model-parallel:

###########################################
Training models with billions of parameters
###########################################

When training large models, fitting larger batch sizes, or trying to increase throughput using multi-GPU compute, Lightning provides advanced optimized distributed training strategies to support these cases and offer substantial improvements in memory usage.
Note that some of the extreme memory saving configurations will affect the speed of training.
This speed/memory trade-off in most cases can be adjusted.

Some of these memory-efficient strategies rely on offloading onto other forms of memory, such as CPU RAM or NVMe.
This means you can even see memory benefits on a **single GPU**, using a strategy such as :ref:`deepspeed-zero-stage-3-offload`.

Check out this amazing video explaining model parallelism and how it works behind the scenes:

.. raw:: html

    <iframe width="540" height="300" src="https://www.youtube.com/embed/w_CKzh5C1K4" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


*****************************************
When NOT to use model-parallel strategies
*****************************************



*********************************************
Choosing the right strategy for your use case
*********************************************

If you've determined that your model is large enough that you need to leverage model parallelism, you have two training strategies to choose from: FSDP, the native solution that comes built-in with PyTorch, or the popular 3rd party `DeepSpeed <https://github.com/microsoft/DeepSpeed>`__ library.
Both have a very similar feature set and have been used to train the largest SOTA models in the world.
Our recommendation is

- **Use FSDP** if you are new to model-parallel training, or if you are migrating from PyTorch FSDP to Lightning.
- **Use DeepSpeed** if you know you will need cutting edge features not present in FSDP, or you are already familiar with DeepSpeed and are migrating to Lightning.

The table below points out a few important differences between the two.

.. list-table:: Differences between FSDP and DeepSpeed
   :header-rows: 1

   * -
     - FSDP
     - DeepSpeed
   * - Dependencies
     - None
     - Requires the ``deepspeed`` package
   * - Configuration options
     - Simple and intuitive
     - Extensive and complex
   * - Configuration
     - Via Trainer
     - Via Trainer or configuration file
   * - Activation checkpointing
     - Yes
     - Yes, but requires changing the model code
   * - Offload parameters
     - CPU
     - CPU or disk
   * - Distributed checkpoints
     - Coming soon
     - Yes


















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


******
Guides
******

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: FSDP
   :description: Distribute models with billions of parameters across hundreds GPUs with FSDP
   :col_css: col-md-4
   :button_link: fsdp.html
   :height: 160
   :tag: advanced

.. displayitem::
   :header: DeepSpeed
   :description: Distribute models with billions of parameters across hundreds GPUs with DeepSpeed
   :col_css: col-md-4
   :button_link: deepspeed.html
   :height: 160
   :tag: advanced


.. raw:: html

        </div>
    </div>
