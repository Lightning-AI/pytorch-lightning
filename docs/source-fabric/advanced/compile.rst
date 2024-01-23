#################################
Speed up models by compiling them
#################################

Compiling your PyTorch model can result in significant speedups, especially on the latest hardware such as NVIDIA GPUs.
This guide shows you how to apply ``torch.compile`` correctly in your code.

.. note::

    This requires PyTorch >= 2.0.


----


***********************************
Apply `torch.compile` to your model
***********************************

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

    # All future executions will be fast
    output = model(input)
    output = model(input)
    ...

This is important to know when you measure the speed of a compiled model and compare it to a regular model.
You should always *exclude* the first call to ``forward()`` from your measurements, since it includes the compilation time.

.. collapse:: Full example with benchmark

    Below is an example that measures the speedup you get when compiling a DenseNet from TorchVision.

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

        model = models.densenet121()
        input = torch.randn(16, 3, 128, 128, device=fabric.device)

        # Compile!
        compiled_model = torch.compile(model, mode="reduce-overhead")

        # Set up the model with Fabric
        model = fabric.setup(model)
        compiled_model = fabric.setup(compiled_model)

        # warmup the compiled model before we benchmark
        compiled_model(input)

        # Run multiple forward passes and time them
        eager_time = benchmark(model, input)
        compile_time = benchmark(compiled_model, input)

        # Compare the speedup for the compiled execution
        speedup = eager_time / compile_time
        print(f"Eager median time: {eager_time:.4f} seconds")
        print(f"Compile median time: {compile_time:.4f} seconds")
        print(f"Speedup: {speedup:.1f}x")

    On an NVIDIA A100 with PyTorch 2.1.2, CUDA 12.1, we get the following speedup:

    .. code-block:: text

        Eager median time: 0.0151 seconds
        Compile median time: 0.0056 seconds
        Speedup: 2.7x


----


******************
Avoid graph breaks
******************

When ``torch.compile`` looks at the code in your model's ``forward()`` method, it will try to compile as much of the code as possible.
If there are regions in the code that it doesn't understand, it will introduce a so-called "graph break" that essentially splits the code in optimized and unoptimized parts.
Graph breaks aren't a deal breaker, since the optimized parts should still run faster.
But if you want to get the most out of ``torch.compile``, you might want to invest rewriting the problematic section of the code that produce the breaks.

You can check whether your model produces graph breaks by calling ``torch.compile`` with ``fullraph=True``:

.. code-block:: python

    # Force an error if there is a graph break in the model
    model = torch.compile(model, fullgraph=True)

The error messages produced here are often quite cryptic.


----


*******************
Avoid recompilation
*******************



----


***********************************
Experiment with compilation options
***********************************



----


*********************************************************
(Experimental) Apply `torch.compile` over FSDP, DDP, etc.
*********************************************************

.. code-block:: python

    # Choose a distributed strategy like DDP or FSDP
    fabric = L.Fabric(devices=2, strategy="ddp")

    # Compile the model
    model = torch.compile(model)

    # Default: `fabric.setup()` will not reapply the compilation over DDP/FSDP
    model = fabric.setup(model, _reapply_compile=False)

    # Recompile the model over DDP/FSDP (experimental)
    model = fabric.setup(model, _reapply_compile=True)
