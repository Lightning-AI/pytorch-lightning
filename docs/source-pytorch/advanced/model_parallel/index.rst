###########################################
Training models with billions of parameters
###########################################

Today, large models with billions of parameters are trained with many GPUs across several machines in parallel.
Even a single H100 GPU with 80 GB of VRAM (one of the biggest today) is not enough to train just a 30B parameter model (even with batch size 1 and 16-bit precision).
The memory consumption for training is generally made up of

1. the model parameters,
2. the layer activations (forward),
3. the gradients (backward),
4. the optimizer states (e.g., Adam has two additional exponential averages per parameter) and
5. model outputs and loss.

|

When the sum of these memory components exceed the VRAM of a single GPU, regular data-parallel training (DDP) can no longer be employed.
To alleviate this limitation, we need to introduce **Model Parallelism**.


----


**************************
What is Model Parallelism?
**************************

There are different types of model parallelism, each with its own trade-offs.

**Fully Sharded Data Parallelism (FSDP)** shards both model parameters and optimizer states across multiple GPUs, significantly reducing memory usage per GPU.
This method, while highly memory-efficient, involves frequent synchronization between GPUs, introducing communication overhead and complexity in implementation.
FSDP is advantageous when memory constraints are the primary issue, provided there are high-bandwidth interconnects to minimize latency.

**Tensor Parallelism (TP)** splits individual tensors across GPUs, enabling fine-grained distribution of computation and memory.
It scales well to a large number of GPUs but requires synchronization of tensor slices after each operation, which adds communication overhead.
TP is most effective with models that have many linear layers (LLMs), offering a balance between memory distribution and computational efficiency.

**Pipeline Parallelism (PP)** divides model layers into segments, each processed by different GPUs, reducing memory load per GPU and minimizing inter-GPU communication to pipeline stage boundaries.
While this reduces communication overhead, it can introduce pipeline bubbles where some GPUs idle, leading to potential inefficiencies.
PP is ideal for deep models with sequential architectures (LLMs), though it requires careful management to minimize idle times.

Choosing a model parallelism style involves considering model architecture, hardware interconnects, and training efficiency.
In practice, hybrid approaches combining FSDP, TP, and PP are often used to leverage the strengths of each method while mitigating their weaknesses.


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

.. displayitem::
    :header: Pipeline Parallelism
    :description: Coming soon
    :col_css: col-md-4
    :height: 180
    :tag: advanced

.. raw:: html

        </div>
    </div>


----


*********************
Parallelisms compared
*********************


**Distributed Data Parallel (DDP)**

.. raw:: html

    <ul class="no-bullets">
        <li>✅ &nbsp; No model code changes required</li>
        <li>✅ &nbsp; Training with very large batch sizes (batch size scales with number of GPUs)</li>
        <li>❗ &nbsp; Model (weights, optimizer state, activations / gradients) must fit into a GPU</li>
    </ul>

|

**Fully-Sharded Data Parallel (FSDP)**

.. raw:: html

    <ul class="no-bullets">
        <li>✅ &nbsp; No model code changes required </li>
        <li>✅ &nbsp; Training with very large batch sizes (batch size scales with number of GPUs) </li>
        <li>✅ &nbsp; Model (weights, optimizer state, gradients) gets distributed across all GPUs </li>
        <li>❗ &nbsp; A single FSDP layer when gathered during forward/backward must fit into the GPU </li>
        <li>❗ &nbsp; Requires some knowledge about model architecture to set configuration options correctly </li>
        <li>❗ &nbsp; Requires very fast networking (multi-node), data transfers between GPUs often become a bottleneck </li>
    </ul>

|

**Tensor Parallel (TP)**

.. raw:: html

    <ul class="no-bullets">
        <li>❗ &nbsp; Model code changes required </li>
        <li>🤔 &nbsp; Fixed global batch size (does not scale with number of GPUs) </li>
        <li>✅ &nbsp; Model (weights, optimizer state, activations) gets distributed across all GPUs</li>
        <li>✅ &nbsp; Parallelizes the computation of layers that are too large to fit onto a single GPU </li>
        <li>❗ &nbsp; Requires lots of knowledge about model architecture to set configuration options correctly </li>
        <li>🤔 &nbsp; Less GPU data transfers required, but data transfers don't overlap with computation like in FSDP </li>
    </ul>

|

**2D Parallel (FSDP + TP)**

.. raw:: html

    <ul class="no-bullets">
        <li>❗ &nbsp; Model code changes required</li>
        <li>✅ &nbsp; Training with very large batch sizes (batch size scales across data-parallel dimension)</li>
        <li>✅ &nbsp; Model (weights, optimizer state, activations) gets distributed across all GPUs</li>
        <li>✅ &nbsp; Parallelizes the computation of layers that are too large to fit onto a single GPU</li>
        <li>❗ &nbsp; Requires lots of knowledge about model architecture to set configuration options correctly</li>
        <li>✅ &nbsp; Tensor-parallel within machines and FSDP across machines reduces data transfer bottlenecks</li>
    </ul>

|

PyTorch Lightning supports all the parallelisms mentioned above natively through PyTorch, with the exception of pipeline parallelism (PP) which is not yet supported.

|
