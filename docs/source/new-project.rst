Quick Start
===========
To start a new project define two files, a LightningModule and a Trainer file.    
To illustrate Lightning power and simplicity, here's an example of a typical research flow.    

Case 1: BERT
------------

Let's say you're working on something like BERT but want to try different ways of training or even different networks.  
You would define a single LightningModule and use flags to switch between your different ideas.

.. code-block:: python

    class BERT(pl.LightningModule):
        def __init__(self, model_name, task):
            self.task = task

            if model_name == 'transformer':
                self.net = Transformer()
            elif model_name == 'my_cool_version':
                self.net = MyCoolVersion()

        def training_step(self, batch, batch_idx):
            if self.task == 'standard_bert':
                # do standard bert training with self.net...
                # return loss

            if self.task == 'my_cool_task':
                # do my own version with self.net
                # return loss


Case 2: COOLER NOT BERT
-----------------------

But if you wanted to try something **completely** different, you'd define a new module for that.


.. code-block:: python

    class CoolerNotBERT(pl.LightningModule):
        def __init__(self):
            self.net = ...

        def training_step(self, batch, batch_idx):
            # do some other cool task
            # return loss


Rapid research flow
-------------------

Then you could do rapid research by switching between these two and using the same trainer.


.. code-block:: python

    if use_bert:
        model = BERT()
    else:
        model = CoolerNotBERT()

    trainer = Trainer(gpus=4, use_amp=True)
    trainer.fit(model)


**Notice a few things about this flow:**

1. You're writing pure PyTorch... no unnecessary abstractions or new libraries to learn.   
2. You get free GPU and 16-bit support without writing any of that code in your model.   
3. You also get all of the capabilities below (without coding or testing yourself).     
