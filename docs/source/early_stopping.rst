Early stopping
==================

Default behavior
----------------------
By default training will go with early stopping if there is
`'val_loss'` in `validation_end()` return dict. Otherwise
training will proceed with early stopping disabled.

.. code-block:: python

    # set None for default behavior
    trainer = Trainer(early_stop_callback=None)


Enable Early Stopping
----------------------
There are two ways to enable early stopping.

.. note:: See: :ref:`trainer`

.. code-block:: python

    # A) Set early_stop_callback to True. Will look for 'val_loss'
    # in validation_end() return dict. If it is not found an error is raised.
    trainer = Trainer(early_stop_callback=True)

    # B) Or configure your own callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )
    trainer = Trainer(early_stop_callback=early_stop_callback)

Disable Early Stopping
-------------------------------------
To disable early stopping pass ``False`` to the `early_stop_callback`.

.. note:: See: :ref:`trainer`

.. code-block:: python

    trainer = Trainer(early_stop_callback=False)