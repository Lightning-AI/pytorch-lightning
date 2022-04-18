.. testsetup:: *

    import torch
    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.core.lightning import LightningModule

.. _gpu:

GPU training (Advanced)
=======================
**Audience:** Users looking to scale massive models (ie: 1 Trillion parameters).

----

For experts pushing the state-of-the-art in model development, Lightning offers various techniques to enable Trillion+ parameter-scale models.

----

.. include:: ../advanced/model_parallel.rst
