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
