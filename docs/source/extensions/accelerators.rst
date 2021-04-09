############
Accelerators
############
Accelerators connect a Lightning Trainer to arbitrary accelerators (CPUs, GPUs, TPUs, etc). Accelerators
also manage distributed communication through :ref:`Plugins` (like DP, DDP, HPC cluster).

Accelerators can also be configured to run on arbitrary clusters using Plugins or to link up to arbitrary
computational strategies like 16-bit precision via AMP and Apex.


.. warning:: The Accelerator API is in beta and subject to change.
    For help setting up custom plugins/accelerators, please reach out to us at **support@pytorchlightning.ai**


----------


Accelerator API
---------------

Accelerator Base Class
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.accelerators.accelerator.Accelerator
   :noindex:


CPU Accelerator
^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.accelerators.cpu.CPUAccelerator
   :noindex:


GPU Accelerator
^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.accelerators.gpu.GPUAccelerator
   :noindex:


TPU Accelerator
^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.accelerators.tpu.TPUAccelerator
   :noindex:
