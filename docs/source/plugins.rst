#######
Plugins
#######

Plugins allow custom integrations to the internals of the Trainer such as a custom amp or ddp implementation.

For example, to customize your own DistributedDataParallel you could do something like this:

.. code-block:: python

    class MyDDP(DDPPlugin):
        ...

    # use your own ddp algorithm
    my_ddp = MyDDP()
    trainer = Trainer(plugins=[my_ddp])

**********
ApexPlugin
**********

.. autoclass:: pytorch_lightning.plugins.apex.ApexPlugin

***************
NativeAMPPlugin
***************

.. autoclass:: pytorch_lightning.plugins.native_amp.NativeAMPPlugin

*********
DDPPlugin
*********

.. autoclass:: pytorch_lightning.plugins.ddp_plugin.DDPPlugin
