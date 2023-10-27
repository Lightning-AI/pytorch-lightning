:orphan:

########################
Own your loop (advanced)
########################

***********************
Customize training loop
***********************

.. image:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/custom_loop.png
    :width: 600
    :alt: Injecting custom code in a training loop

Inject custom code anywhere in the Training loop using any of the 20+ methods (:ref:`lightning_hooks`) available in the LightningModule.

.. testcode::

    import lightning as L


    class LitModel(L.LightningModule):
        def backward(self, loss):
            loss.backward()

----

.. include:: manual_optimization.rst
