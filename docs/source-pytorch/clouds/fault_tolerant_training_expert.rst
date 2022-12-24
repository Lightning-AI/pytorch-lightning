:orphan:

################################
Fault-tolerant Training (expert)
################################
**Audience**: Experts looking to enable and handle their own fault-tolerance.

**Pre-requisites**: Users must have first read :doc:`Fault-tolrance Training (basic) <fault_tolerant_training_basic>`

----

***************************************
Enable fault-tolerant behavior anywhere
***************************************
To enable fault tolerance on your own cloud or cluster environment enable the *PL_FAULT_TOLERANT_TRAINING* environment variable:

.. code-block:: bash

    PL_FAULT_TOLERANT_TRAINING=1 python script.py

Although Lighting will now be fault-tolerant, you'll have to handle all the nuances of making sure the models are automatically restarted.

.. note:: This complexity is already handled for you if you use **lightning-grid**.
