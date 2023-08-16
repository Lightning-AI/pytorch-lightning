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


----


*****************************************
When NOT to use model-parallel strategies
*****************************************

Model parallel techniques help when model sizes are fairly large; roughly 500M+ parameters is where we've seen benefits.
For small models (for example ResNet50 of around 80M Parameters) where the weights, activations, optimizer states and gradients all fit in GPU memory, you do not need to use a model-parallel strategy.
Instead, use regular :ref:`distributed data-parallel (DDP) <gpu_intermediate>` training to scale your batch size and speed up training across multiple GPUs and machines.
There are several :ref:`DDP optimizations <ddp-optimizations>` you can explore if memory and speed are a concern.


----


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
     - :ref:`FSDP <fully-sharded-training>`
     - :ref:`DeepSpeed <deepspeed_advanced>`
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


----


***********
Get started
***********

Once you've chosen the right strategy for your use case, follow the full guide below to get started.

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


----


**********************
Third-party strategies
**********************

Cutting-edge Lightning strategies are being developed by third-parties outside of Lightning.
If you want to try some of the latest and greatest features for model-parallel training, check out these integrations:

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Colossal-AI
   :description: Has advanced distributed training algorithms and system optimizations
   :col_css: col-md-4
   :button_link: ../integrations/strategies/colossalai.html
   :height: 160
   :tag: advanced

.. displayitem::
   :header: Bagua
   :description: Has advanced distributed training algorithms and system optimizations
   :col_css: col-md-4
   :button_link: ../integrations/strategies/bagua.html
   :height: 160
   :tag: advanced

.. displayitem::
   :header: Hivemind
   :description: For training on unreliable mixed GPUs across the internet
   :col_css: col-md-4
   :button_link: ../integrations/strategies/hivemind.html
   :height: 160
   :tag: advanced


.. raw:: html

        </div>
    </div>
