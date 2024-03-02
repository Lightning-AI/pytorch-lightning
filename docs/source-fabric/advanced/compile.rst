#################################
Speed up models by compiling them
#################################

Compiling your PyTorch model can result in significant speedups, especially on the latest generations of GPUs.
This guide shows you how to apply `torch.compile <https://pytorch.org/docs/2.2/generated/torch.compile.html>`_ correctly in your code.

.. note::

    This requires PyTorch >= 2.0.


----


*********************************
Apply torch.compile to your model
*********************************

Compiling a model in a script together with Fabric is as simple as adding one line of code, calling :func:`torch.compile`:

.. code-block:: python

    import torch
    import lightning as L

    # Set up Fabric
    fabric = L.Fabric(devices=1)

    # Define the model
    model = ...

    # Compile the model
    model = torch.compile(model)

    # `fabric.setup()` should come after `torch.compile()`
    model = fabric.setup(model)


.. important::

    You should compile the model **before** calling ``fabric.setup()`` as shown above for an optimal integration with features in Fabric.

The newly added call to ``torch.compile()`` by itself doesn't do much. It just wraps the model in a "compiled model".
The actual optimization will start when calling ``forward()`` on the model for the first time:

.. code-block:: python

    # 1st execution compiles the model (slow)
    output = model(input)

    # All future executions will be fast (for inputs of the same size)
    output = model(input)
    output = model(input)
    ...


When measuring the speed of a compiled model and comparing it to a regular model, it is important to
always exclude the first call to ``forward()`` from your measurements, since it includes the compilation time.


.. collapse:: Full example with benchmark

    Below is an example that measures the speedup you get when compiling the InceptionV3 from TorchVision.

    .. code-block:: python

        import statistics
        import torch
        import torchvision.models as models
        import lightning as L


        @torch.no_grad()
        def benchmark(model, input, num_iters=10):
            """Runs the model on the input several times and returns the median execution time."""
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            times = []
            for _ in range(num_iters):
                start.record()
                model(input)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end) / 1000)
            return statistics.median(times)


        fabric = L.Fabric(accelerator="cuda", devices=1)

        model = models.inception_v3()
        input = torch.randn(16, 3, 510, 512, device=fabric.device)

        # Compile!
        compiled_model = torch.compile(model)

        # Set up the model with Fabric
        model = fabric.setup(model)
        compiled_model = fabric.setup(compiled_model)

        # warm up the compiled model before we benchmark
        compiled_model(input)

        # Run multiple forward passes and time them
        eager_time = benchmark(model, input)
        compile_time = benchmark(compiled_model, input)

        # Compare the speedup for the compiled execution
        speedup = eager_time / compile_time
        print(f"Eager median time: {eager_time:.4f} seconds")
        print(f"Compile median time: {compile_time:.4f} seconds")
        print(f"Speedup: {speedup:.1f}x")

    On an NVIDIA A100 SXM4 40GB with PyTorch 2.2.0, CUDA 12.1, we get the following speedup:

    .. code-block:: text

        Eager median time: 0.0254 seconds
        Compile median time: 0.0185 seconds
        Speedup: 1.4x


----


******************
Avoid graph breaks
******************

When ``torch.compile`` looks at the code in your model's ``forward()`` method, it will try to compile as much of the code as possible.
If there are regions in the code that it doesn't understand, it will introduce a so-called "graph break" that essentially splits the code in optimized and unoptimized parts.
Graph breaks aren't a deal breaker, since the optimized parts should still run faster.
But if you want to get the most out of ``torch.compile``, you might want to invest rewriting the problematic section of the code that produce the breaks.

You can check whether your model produces graph breaks by calling ``torch.compile`` with ``fullgraph=True``:

.. code-block:: python

    # Force an error if there is a graph break in the model
    model = torch.compile(model, fullgraph=True)

Be aware that the error messages produced here are often quite cryptic, so you will likely have to do some `troubleshooting <https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html>`_ to fully optimize your model.


----


*******************
Avoid recompilation
*******************

As mentioned before, the compilation of the model happens the first time you call ``forward()``.
At this point, PyTorch will inspect the input tensor(s) and optimize the compiled code for the particular shape, data type and other properties the input has.
If the shape of the input remains the same across all calls to ``forward()``, PyTorch will reuse the compiled code it generated and you will get the best speedup.
However, if these properties change across subsequent calls to ``forward()``, PyTorch will be forced to recompile the model for the new shapes, and this will significantly slow down your training if it happens on every iteration.

**When your training suddenly becomes slow, it's probably because PyTorch is recompiling the model!**
Here are some common scenarios when this can happen:

- Your training code includes an evaluation step on a different dataset, or you are using a ``Trainer`` that switches from training to validation/testing and the input shape changes, triggering a recompilation.
- Your dataset size is not divisible by the batch size, and the dataloader has ``drop_last=False`` (the default).
  The last batch in your training loop will be smaller and trigger a recompilation.

Ideally, you should try to make the input shape(s) to ``forward()`` static.
However, when this is not possible, you can request PyTorch to compile the code by taking into account possible changes to the input shapes.

.. code-block:: python

    # On PyTorch < 2.2
    model = torch.compile(model, dynamic=True)

A model compiled with ``dynamic=True`` will typically be slower than a model compiled with static shapes, but it will avoid the extreme cost of recompilation every iteration.
On PyTorch 2.2 and later, ``torch.compile`` will detect dynamism automatically and you should no longer need to set this.

.. collapse:: Example with dynamic shapes

    The code below shows an example where the model recompiles for several seconds because the input shape changed.
    You can compare the timing results by toggling ``dynamic=True/False`` in the call to ``torch.compile``:

    .. code-block:: python

        import time
        import torch
        import torchvision.models as models
        import lightning as L

        fabric = L.Fabric(accelerator="cuda", devices=1)

        model = models.inception_v3()

        # dynamic=False is the default
        torch._dynamo.config.automatic_dynamic_shapes = False

        compiled_model = torch.compile(model)
        compiled_model = fabric.setup(compiled_model)

        input = torch.randn(16, 3, 512, 512, device=fabric.device)
        t0 = time.time()
        compiled_model(input)
        torch.cuda.synchronize()
        print(f"1st forward: {time.time() - t0:.2f} seconds.")

        input = torch.randn(8, 3, 512, 512, device=fabric.device)  # note the change in shape
        t0 = time.time()
        compiled_model(input)
        torch.cuda.synchronize()
        print(f"2nd forward: {time.time() - t0:.2f} seconds.")

    With ``automatic_dynamic_shapes=True``:

    .. code-block:: text

        1st forward: 41.90 seconds.
        2nd forward: 89.27 seconds.

    With ``automatic_dynamic_shapes=False``:

    .. code-block:: text

        1st forward: 42.12 seconds.
        2nd forward: 47.77 seconds.

    Numbers produced with NVIDIA A100 SXM4 40GB, PyTorch 2.2.0, CUDA 12.1.


If you still see recompilation issues after dealing with the aforementioned cases, there is a `Compile Profiler in PyTorch <https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html#excessive-recompilation>`_ for further investigation.


----


***********************************
Experiment with compilation options
***********************************

There are optional settings that, depending on your model, can give additional speedups.

**CUDA Graphs:** By enabling CUDA Graphs, CUDA will record all computations in a graph and replay it every time forward and backward is called.
The requirement is that your model must be static, i.e., the input shape must not change and your model must execute the same operations every time.
Enabling CUDA Graphs often results in a significant speedup, but sometimes also increases the memory usage of your model.

.. code-block:: python

    # Enable CUDA Graphs
    compiled_model = torch.compile(model, mode="reduce-overhead")

    # This does the same
    compiled_model = torch.compile(model, options={"triton.cudagraphs": True})

|

**Shape padding:** The specific shape/size of the tensors involved in the computation of your model (input, activations, weights, gradients, etc.) can have an impact on the performance.
With shape padding enabled, ``torch.compile`` can extend the tensors by padding to a size that gives a better memory alignment.
Naturally, the tradoff here is that it will consume a bit more memory.

.. code-block:: python

    # Default is False
    compiled_model = torch.compile(model, options={"shape_padding": True})


You can find a full list of compile options in the `PyTorch documentation <https://pytorch.org/docs/stable/generated/torch.compile.html>`_.


----


**************************************
A note about torch.compile in practice
**************************************

In practice, you will find that ``torch.compile`` may not work well at first or may be counter-productive to performance.
Compilation may fail with cryptic error messages that are hard to debug, luckily the PyTorch team is responsive and it's likely that messaging will improve in time.
It is not uncommon that ``torch.compile`` will produce a significantly *slower* model or one with higher memory usage. You'll need to invest time in this phase if the model is not among the ones that have a happy path.
As a note, the compilation phase itself will take some time, taking up to several minutes.
For these reasons, we recommend that you don't invest too much time trying to apply ``torch.compile`` during development, and rather evaluate its effectiveness toward the end when you are about to launch long-running, expensive experiments.
Always compare the speed and memory usage of the compiled model against the original model!


----


*************************************
Using torch.compile with FSDP and DDP
*************************************

As stated earlier, we recommend that you compile the model before calling ``fabric.setup()``.
In the case of DDP and FSDP, ``fabric.setup()`` will automatically reapply the ``torch.compile`` call after the model gets wrapped in DDP/FSDP internally.
This will ensure that the compilation can incorporate the distributed calls and optimize them.
However, should you have issues compiling DDP and FSDP models, you can opt out of this feature:

.. code-block:: python

    # Choose a distributed strategy like DDP or FSDP
    fabric = L.Fabric(devices=2, strategy="ddp")

    # Compile the model
    model = torch.compile(model)

    # Default: `fabric.setup()` will configure compilation over DDP/FSDP for you
    model = fabric.setup(model, _reapply_compile=True)

    # Turn it off if you see issues with DDP/FSDP
    model = fabric.setup(model, _reapply_compile=False)


----


********************
Additional Resources
********************

Here are a few resources for further reading after you complete this tutorial:

- `PyTorch 2.0 Paper <https://pytorch.org/blog/pytorch-2-paper-tutorial/>`_
- `GenAI with PyTorch 2.0 blog post series <https://pytorch.org/blog/accelerating-generative-ai-4/>`_
- `Training Production AI Models with PyTorch 2.0 <https://pytorch.org/blog/training-production-ai-models/>`_
- `Empowering Models with Performance: The Art of Generalized Model Transformation Approach <https://pytorch.org/blog/empowering-models-performance/>`_

|
