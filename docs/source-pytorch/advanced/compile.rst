#################################
Speed up models by compiling them
#################################

Compiling your LightningModule can result in significant speedups, especially on the latest generations of GPUs.
This guide shows you how to apply `torch.compile <https://pytorch.org/docs/2.2/generated/torch.compile.html>`_ correctly in your code.

.. note::

    This requires PyTorch >= 2.0.


----


*******************************************
Apply torch.compile to your LightningModule
*******************************************

Compiling a LightningModule is as simple as adding one line of code, calling :func:`torch.compile`:

.. code-block:: python

    import torch
    import lightning as L

    # Define the model
    model = MyLightningModule()

    # Compile the model
    model = torch.compile(model)

    # Run with the Trainer
    trainer = L.Trainer()
    trainer.fit(model)


.. important::

    You should compile the model **before** calling ``trainer.fit()`` as shown above for an optimal integration with features in Trainer.

The newly added call to ``torch.compile()`` by itself doesn't do much. It just wraps the model in a "compiled model".
The actual optimization will start when calling the ``forward()`` method for the first time:

.. code-block:: python

    # 1st execution compiles the model (slow)
    output = model(input)

    # All future executions will be fast (for inputs of the same size)
    output = model(input)
    output = model(input)
    ...

**When you pass the LightningModule to the Trainer, it will automatically also compile the ``*_step()`` methods.**

When measuring the speed of a compiled model and comparing it to a regular model, it is important to
always exclude the first call to ``forward()``/``*_step()`` from your measurements, since it includes the compilation time.


.. collapse:: Full example with benchmark

    Below is an example that measures the speedup you get when compiling the InceptionV3 from TorchVision.

    .. code-block:: python

        import statistics
        import torch
        import torchvision.models as models
        import lightning as L
        from torch.utils.data import DataLoader


        class MyLightningModule(L.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = models.inception_v3()

            def training_step(self, batch):
                return self.model(batch).logits.sum()

            def train_dataloader(self):
                return DataLoader([torch.randn(3, 512, 512) for _ in range(256)], batch_size=16)

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.01)


        class Benchmark(L.Callback):
            """A callback that measures the median execution time between the start and end of a batch."""
            def __init__(self):
                self.start = torch.cuda.Event(enable_timing=True)
                self.end = torch.cuda.Event(enable_timing=True)
                self.times = []

            def median_time(self):
                return statistics.median(self.times)

            def on_train_batch_start(self, trainer, *args, **kwargs):
                self.start.record()

            def on_train_batch_end(self, trainer, *args, **kwargs):
                # Exclude the first iteration to let the model warm up
                if trainer.global_step > 1:
                    self.end.record()
                    torch.cuda.synchronize()
                    self.times.append(self.start.elapsed_time(self.end) / 1000)


        model = MyLightningModule()

        # Compile!
        compiled_model = torch.compile(model)

        # Measure the median iteration time with uncompiled model
        benchmark = Benchmark()
        trainer = L.Trainer(accelerator="cuda", devices=1, max_steps=10, callbacks=[benchmark])
        trainer.fit(model)
        eager_time = benchmark.median_time()

        # Measure the median iteration time with compiled model
        benchmark = Benchmark()
        trainer = L.Trainer(accelerator="cuda", devices=1, max_steps=10, callbacks=[benchmark])
        trainer.fit(compiled_model)
        compile_time = benchmark.median_time()

        # Compare the speedup for the compiled execution
        speedup = eager_time / compile_time
        print(f"Eager median time: {eager_time:.4f} seconds")
        print(f"Compile median time: {compile_time:.4f} seconds")
        print(f"Speedup: {speedup:.1f}x")


    On an NVIDIA A100 SXM4 40GB with PyTorch 2.2.0, CUDA 12.1, we get the following speedup:

    .. code-block:: text

        Eager median time: 0.0863 seconds
        Compile median time: 0.0709 seconds
        Speedup: 1.2x


----


******************
Avoid graph breaks
******************

When ``torch.compile`` looks at the code in your model's ``forward()`` or ``*_step()`` method, it will try to compile as much of the code as possible.
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

As mentioned before, the compilation of the model happens the first time you call ``forward()`` or the first time the Trainer calls the ``*_step()`` methods.
At this point, PyTorch will inspect the input tensor(s) and optimize the compiled code for the particular shape, data type and other properties the input has.
If the shape of the input remains the same across all calls, PyTorch will reuse the compiled code it generated and you will get the best speedup.
However, if these properties change across subsequent calls to ``forward()``/``*_step()``, PyTorch will be forced to recompile the model for the new shapes, and this will significantly slow down your training if it happens on every iteration.

**When your training suddenly becomes slow, it's probably because PyTorch is recompiling the model!**
Here are some common scenarios when this can happen:

- You are using dataset with different inputs or shapes for validation than for training, causing a recompilation whenever the Trainer switches between training and validation.
- Your dataset size is not divisible by the batch size, and the dataloader has ``drop_last=False`` (the default).
  The last batch in your training loop will be smaller and trigger a recompilation.

Ideally, you should try to make the input shape(s) to ``forward()`` static.
However, when this is not possible, you can request PyTorch to compile the code by taking into account possible changes to the input shapes.

.. code-block:: python

    # On PyTorch < 2.2
    model = torch.compile(model, dynamic=True)

A model compiled with ``dynamic=True`` will typically be slower than a model compiled with static shapes, but it will avoid the extreme cost of recompilation every iteration.
On PyTorch 2.2 and later, ``torch.compile`` will detect dynamism automatically and you should no longer need to set this.

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


***********
Limitations
***********

There are a few limitations you should be aware of when using ``torch.compile`` **in conjunction with the Trainer**:

* The Trainer currently does not reapply ``torch.compile`` over DDP/FSDP, meaning distributed operations can't benefit from speed ups at the moment.
  This limitation will be lifted in the future.

* In some cases, using ``self.log()`` in your LightningModule will cause compilation errors.
  Until addressed, you can work around these issues by applying ``torch.compile`` to the submodule(s) of your LightningModule rather than to the entire LightningModule at once.

  .. code-block:: python

      import lightning as L

      class MyLightningModule(L.LightningModule):
          def __init__(self):
              super().__init__()
              self.model = MySubModule()
              self.model = torch.compile(self.model)
              ...


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
