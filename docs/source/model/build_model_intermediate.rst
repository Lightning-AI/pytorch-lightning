:orphan:

###################################
Supercharge training (intermediate)
###################################

************************
Enable training features
************************
Enable advanced training features using Trainer arguments. These are SOTA techniques that are automatically integrated into your training loop without changes to your code.

.. code::

   # train 1TB+ parameter models with Deepspeed/fsdp
   trainer = Trainer(
       devices=4,
       accelerator="gpu",
       strategy="deepspeed_stage_2",
       precision=16
    )

   # 20+ helpful arguments for rapid idea iteration
   trainer = Trainer(
       max_epochs=10,
       min_epochs=5,
       overfit_batches=1
    )

   # access the latest state of the art techniques
   trainer = Trainer(callbacks=[StochasticWeightAveraging(...)])

----

******************
Extend the Trainer
******************

.. raw:: html

    <video width="100%" max-width="800px" controls autoplay muted playsinline
    src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/cb.m4v"></video>

If you have multiple lines of code with similar functionalities, you can use *callbacks* to easily group them together and toggle all of those lines on or off at the same time.

.. code::

   trainer = Trainer(callbacks=[AWSCheckpoints()])
