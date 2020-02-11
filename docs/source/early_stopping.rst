Early stopping
==================


Enable Early Stopping
----------------------
There are two ways to enable early stopping.

.. note:: See: :ref:`trainer`

.. code-block:: python

    # A) Looks for val_loss in validation_step return dict
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

Force disable early stop
-------------------------------------
To disable early stopping pass None to the early_stop_callback

.. note:: See: :ref:`trainer`

.. code-block:: python

   # DEFAULT
   trainer = Trainer(early_stop_callback=None)