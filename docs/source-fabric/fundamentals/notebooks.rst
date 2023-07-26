.. _Fabric in Notebooks:

###################
Fabric in Notebooks
###################

Fabric works the same way in notebooks (Jupyter, Google Colab, Kaggle, etc.) if you only run in a single process or GPU.
If you want to use multiprocessing, for example, multi-GPU, you can put your code in a function and pass that function to the
:meth:`~lightning.fabric.fabric.Fabric.launch` method:


.. code-block:: python


    # Notebook Cell
    def train(fabric):
        model = ...
        optimizer = ...
        model, optimizer = fabric.setup(model, optimizer)
        ...


    # Notebook Cell
    fabric = Fabric(accelerator="cuda", devices=2)
    fabric.launch(train)  # Launches the `train` function on two GPUs


As you can see, this function accepts one argument, the ``Fabric`` object, and it gets launched on as many devices as specified.


----


*********************
Multi-GPU Limitations
*********************

The multi-GPU capabilities in Jupyter are enabled by launching processes using the 'fork' start method.
It is the only supported way of multi-processing in notebooks, but also brings some limitations that you should be aware of.

Avoid initializing CUDA before launch
=====================================

Don't run torch CUDA functions before calling ``fabric.launch(train)`` in any of the notebook cells beforehand, otherwise your code may hang or crash.

.. code-block:: python

    # BAD: Don't run CUDA-related code before `.launch()`
    # x = torch.tensor(1).cuda()
    # torch.cuda.empty_cache()
    # torch.cuda.is_available()


    def train(fabric):
        # GOOD: Move CUDA calls into the training function
        x = torch.tensor(1).cuda()
        torch.cuda.empty_cache()
        torch.cuda.is_available()
        ...


    fabric = Fabric(accelerator="cuda", devices=2)
    fabric.launch(train)


Move data loading code inside the function
==========================================

If you define/load your data in the main process before calling ``fabric.launch(train)``, you may see a slowdown or crashes (segmentation fault, SIGSEV, etc.).
The best practice is to move your data loading code inside the training function to avoid these issues:

.. code-block:: python

    # BAD: Don't load data in the main process
    # dataset = MyDataset("data/")
    # dataloader = torch.utils.data.DataLoader(dataset)


    def train(fabric):
        # GOOD: Move data loading code into the training function
        dataset = MyDataset("data/")
        dataloader = torch.utils.data.DataLoader(dataset)
        ...


    fabric = Fabric(accelerator="cuda", devices=2)
    fabric.launch(train)
