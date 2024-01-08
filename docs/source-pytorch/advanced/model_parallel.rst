.. _model-parallel:

########################################
Train models with billions of parameters
########################################

**Audience**: Users who want to train massive models of billions of parameters efficiently across multiple GPUs and machines.

Lightning provides advanced and optimized model-parallel training strategies to support massive models of billions of parameters.
Check out this amazing video for an introduction to model parallelism and its benefits:

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

If you've determined that your model is large enough that you need to leverage model parallelism, you have two training strategies to choose from: FSDP, the native solution that comes built-in with PyTorch, or the popular third-party `DeepSpeed <https://github.com/microsoft/DeepSpeed>`__ library.
Both have a very similar feature set and have been used to train the largest SOTA models in the world.
Our recommendation is

- **Use FSDP** if you are new to model-parallel training, if you are migrating from PyTorch FSDP to Lightning, or if you are already familiar with DDP.
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
     - Simpler and easier to get started
     - More comprehensive, allows finer control
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
   :button_link: model_parallel/fsdp.html
   :height: 160
   :tag: advanced

.. displayitem::
   :header: DeepSpeed
   :description: Distribute models with billions of parameters across hundreds GPUs with DeepSpeed
   :col_css: col-md-4
   :button_link: model_parallel/deepspeed.html
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
If you want to try some of the latest and greatest features for model-parallel training, check out these :doc:`strategies <../integrations/strategies/index>`.
