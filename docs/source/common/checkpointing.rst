.. testsetup:: *

    import os
    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.core.lightning import LightningModule

.. _checkpointing:

#############
Checkpointing
#############

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Save and restore model progress
   :description: Learn to save and load checkpoints
   :col_css: col-md-3
   :button_link: checkpointing_basic.html
   :height: 150
   :tag: basic

.. displayitem::
   :header: Control checkpointing rules
   :description: Customize checkpointing behavior
   :col_css: col-md-3
   :button_link: checkpointing_intermediate.html
   :height: 150
   :tag: intermediate

.. displayitem::
   :header: Save checkpoints to the cloud
   :description: Enable cloud-based checkpointing and composable checkpoints.
   :col_css: col-md-3
   :button_link: checkpointing_advanced.html
   :height: 150
   :tag: advanced

.. displayitem::
   :header: Modify how checkpoints are saved
   :description: Customize checkpointing for custom distributed strategies and accelerators.
   :col_css: col-md-3
   :button_link: checkpointing_expert.html
   :height: 150
   :tag: expert

.. raw:: html

        </div>
    </div>