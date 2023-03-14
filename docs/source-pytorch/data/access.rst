:orphan:

Accessing DataLoaders
=====================

In the case that you require access to the :class:`torch.utils.data.DataLoader` or :class:`torch.utils.data.Dataset` objects, DataLoaders for each step can be accessed
via the trainer properties :meth:`~lightning.pytorch.trainer.trainer.Trainer.train_dataloader`,
:meth:`~lightning.pytorch.trainer.trainer.Trainer.val_dataloaders`,
:meth:`~lightning.pytorch.trainer.trainer.Trainer.test_dataloaders`, and
:meth:`~lightning.pytorch.trainer.trainer.Trainer.predict_dataloaders`.

.. code-block:: python

    dataloaders = trainer.train_dataloader
    dataloaders = trainer.val_dataloaders
    dataloaders = trainer.test_dataloaders
    dataloaders = trainer.predict_dataloaders

These properties will match exactly what was returned in your ``*_dataloader`` hooks or passed to the ``Trainer``,
meaning that if you returned a dictionary of dataloaders, these will return a dictionary of dataloaders.

Replacing DataLoaders
---------------------

If you are using a :class:`~lightning.pytorch.utilities.CombinedLoader`. A flattened list of DataLoaders can be accessed by doing:

.. code-block:: python

    from lightning.pytorch.utilities import CombinedLoader

    iterables = {"dl1": dl1, "dl2": dl2}
    combined_loader = CombinedLoader(iterables)
    # access the original iterables
    assert combined_loader.iterables is iterables
    # the `.flattened` property can be convenient
    assert combined_loader.flattened == [dl1, dl2]
    # for example, to do a simple loop
    updated = []
    for dl in combined_loader.flattened:
        new_dl = apply_some_transformation_to(dl)
        updated.append(new_dl)
    # it also allows you to easily replace the dataloaders
    combined_loader.flattened = updated
