:orphan:

#################################
Introduction to Model Parallelism
#################################

Model parallelism is a technique used in deep learning to train very large models across multiple devices, making it possible to handle models that are too big for a single device's memory.
This is important as models become increasingly complex and large.

Model parallelism comes into play when a model's parameters are too large for one GPU.
There are different types of parallelism, each with its own advantages and trade-offs.
**Data Parallelism (DDP)** divides the data across multiple devices, with each device having a complete copy of the model and processing different parts of the data.
It's easy to implement but doesn't solve memory issues for very large models.
**Fully Sharded Data Parallel (FSDP)** improves on this by spreading the model weights across GPUs, which saves memory but makes communication between devices more complex.
**Tensor Parallelism (TP)** breaks down individual layers of the model to spread them across devices, allowing for more detailed distribution but needing careful synchronization.
**Pipeline Parallelism (PP)** splits the model into stages, with each stage assigned to a different device.
This reduces memory use per device but can introduce delays and inefficiencies (pipeline bubbles).
Choosing the right type of model parallelism depends on balancing memory use, communication needs, and computational efficiency, based on the specific model and hardware setup.


----


*********************
Parallelisms compared
*********************


**Distributed Data Parallel (DDP)**

.. raw:: html

    <ul class="no-bullets">
        <li>✅No model code changes required</li>
        <li>✅Training with very large batch sizes (batch size scales with number of GPUs)</li>
        <li>❗Model (weights, optimizer state, activations) must fit into a GPU</li>
    </ul>

|

**Fully-Sharded Data Parallel (FSDP)**

.. raw:: html

    <ul class="no-bullets">
        <li>✅No model code changes required </li>
        <li>✅Training with very large batch sizes (batch size scales with number of GPUs)</li>
        <li>✅Model (weights, optimizer state, activations) gets distributed across all GPUs </li>
        <li>❗Forward/backward computation requires a single layer must fit into a GPU </li>
        <li>❗Requires some knowledge about model architecture to set configuration options correctly </li>
        <li>❗Requires very fast networking (multi-node), data transfers between GPUs often become a bottleneck </li>
    </ul>

|

**Tensor Parallel (TP)**

.. raw:: html

    <ul class="no-bullets">
        <li>❗Model code changes required </li>
        <li>🤔Fixed global batch size (does not scale with number of GPUs) </li>
        <li>✅Model (weights, optimizer state, activations) gets distributed across all GPUs</li>
        <li>✅Parallelizes the computation of layers that are too large to fit onto a single GPU </li>
        <li>❗Requires lots of knowledge about model architecture to set configuration options correctly </li>
        <li>🤔Less GPU data transfers required, but data transfers don't overlap with computation like in FSDP </li>
    </ul>

|

**2D Parallel (FSDP + TP)**

.. raw:: html

    <ul class="no-bullets">
        <li>❗Model code changes required</li>
        <li>✅Training with very large batch sizes (batch size scales across data-parallel dimension)</li>
        <li>✅Model (weights, optimizer state, activations) gets distributed across all GPUs</li>
        <li>✅Parallelizes the computation of layers that are too large to fit onto a single GPU</li>
        <li>❗Requires lots of knowledge about model architecture to set configuration options correctly</li>
        <li>✅Tensor-parallel within machines and FSDP across machines reduces data transfer bottlenecks</li>
    </ul>

|

Lightning Fabric supports all the parallelisms mentioned above natively through PyTorch, with the exception of pipeline parallelism (PP) which is not yet supported.



----


***********
Get started
***********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: Fully-Sharded Data Parallel (FSDP)
    :description: Get started training large multi-billion parameter models with minimal code changes
    :col_css: col-md-4
    :button_link: fsdp.html
    :height: 180
    :tag: advanced

.. displayitem::
    :header: Tensor Parallel (TP)
    :description: Learn the principles behind tensor parallelism and how to apply it to your model
    :col_css: col-md-4
    :button_link: tp.html
    :height: 180
    :tag: advanced

.. displayitem::
    :header: 2D Parallel (FSDP + TP)
    :description: Combine Tensor Parallelism with FSDP (2D Parallel) to train efficiently on 100s of GPUs
    :button_link: tp_fsdp.html
    :col_css: col-md-4
    :height: 180
    :tag: advanced

.. raw:: html

        </div>
    </div>

|
