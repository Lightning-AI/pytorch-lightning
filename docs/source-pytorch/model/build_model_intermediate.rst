:orphan:

###################################
Supercharge training (intermediate)
###################################

************************
Enable training features
************************
Enable advanced training features using Trainer arguments. These are SOTA techniques that are automatically integrated into your training loop without changes to your code.

.. code::

   # train 1T+ parameter models with DeepSpeed/FSDP
   trainer = Trainer(
       devices=4,
       accelerator="gpu",
       strategy="deepspeed_stage_2",
       precision="16-mixed",
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

.. video:: https://pl-public-data.s3.amazonaws.com/assets_lightning/cb.mp4
    :width: 600
    :autoplay:
    :loop:
    :muted:

If you have multiple lines of code with similar functionalities, you can use *callbacks* to easily group them together and toggle all of those lines on or off at the same time.

.. code::

   trainer = Trainer(callbacks=[AWSCheckpoints()])
