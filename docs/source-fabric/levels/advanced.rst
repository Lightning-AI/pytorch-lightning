.. toctree::
   :maxdepth: 1
   :hidden:

    <../advanced/gradient_accumulation>
    <../advanced/distributed_communication>
    <../advanced/multiple_setup>
    <../advanced/compile>
    <../advanced/model_parallel/fsdp>
    <../guide/checkpoint/distributed_checkpoint>


###############
Advanced skills
###############

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: Use efficient gradient accumulation
    :description: Learn how to perform efficient gradient accumulation in distributed settings
    :button_link: ../advanced/gradient_accumulation.html
    :col_css: col-md-4
    :height: 170
    :tag: advanced

.. displayitem::
    :header: Distribute communication
    :description: Learn all about communication primitives for distributed operation. Gather, reduce, broadcast, etc.
    :button_link: ../advanced/distributed_communication.html
    :col_css: col-md-4
    :height: 170
    :tag: advanced

.. displayitem::
    :header: Use multiple models and optimizers
    :description: See how flexible Fabric is to work with multiple models and optimizers!
    :button_link: ../advanced/multiple_setup.html
    :col_css: col-md-4
    :height: 170
    :tag: advanced

.. displayitem::
    :header: Speed up models by compiling them
    :description: Use torch.compile to speed up models on modern hardware
    :button_link: ../advanced/compile.html
    :col_css: col-md-4
    :height: 170
    :tag: advanced

.. displayitem::
    :header: Train models with billions of parameters
    :description: Train the largest models with FSDP/TP across multiple GPUs and machines
    :button_link: ../advanced/model_parallel/index.html
    :col_css: col-md-4
    :height: 170
    :tag: advanced

.. displayitem::
    :header: Save and load very large models
    :description: Save and load very large models efficiently with distributed checkpoints
    :button_link: ../guide/checkpoint/distributed_checkpoint.html
    :col_css: col-md-4
    :height: 170
    :tag: advanced

.. raw:: html

        </div>
    </div>
