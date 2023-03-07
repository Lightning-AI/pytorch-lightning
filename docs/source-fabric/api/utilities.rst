################
Fabric Utilities
################


seed_everything
===============

This function sets the random seed in important libraries.
In a single line of code, you can seed PyTorch, NumPy, and Python:

.. code-block:: diff

    + from lightning.fabric import seed_everything

      seed = 42
    - random.seed(seed)
    - numpy.random.seed(seed)
    - torch.manual_seed(seed)
    - torch.cuda.manual_seed(seed)

    + seed_everything(seed)

The same is also available as a method on the Fabric object if you don't want to import it separately:

.. code-block:: python

    from lightning.fabric import Fabric

    fabric.Fabric()
    fabric.seed_everything(42)


In distributed settings, you may need to set a different seed per process, depending on the application.
For example, when generating noise or data augmentations. This is very straightforward:

.. code-block:: python

    fabric = Fabric(...)
    fabric.seed_everything(seed + fabric.global_rank)


By default, :meth:`~lightning.fabric.fabric.Fabric.seed_everything` also handles the initialization of the seed in :class:`~torch.utils.data.DataLoader` worker processes:

.. code-block:: python

    fabric = Fabric(...)

    # By default, we handle DataLoader workers too:
    fabric.seed_everything(..., workers=True)

    # Can be turned off:
    fabric.seed_everything(..., workers=False)


----


print
=====

Avoid duplicated print statements in the logs in distributed training by using Fabric's :meth:`~lightning.fabric.fabric.Fabric.print` method:

.. code-block:: python

    print("This message gets printed in every process. That's a bit messy!")

    fabric = Fabric(...)
    fabric.print("This message gets printed only in the main process. Much cleaner!")
