#######################
Run an App on the Cloud
#######################

**Audience:** Users who want to share their apps or run on specialized hardware (like GPUs).

----

*********************************
Run on the public Lightning cloud
*********************************
To run any app on the public lightning cloud use the ``--cloud`` argument:

.. code:: bash

    lightning_app run app app.py --cloud


.. note::
    By default, running your apps on the public Lightning cloud is free of charge using default CPUs, and any app uploaded to the Lightning cloud will be shared with the community (source code and app view will be public). If you would like to make your apps private please `contact us <mailto:support@lightning.ai?subject=I%20want%20private%20apps!>`_.

If your app contains ``LightningWork`` components that require more compute resources, such as larger CPUs or **GPUs**, you'll need to add credits to your Lightning AI account.


----

**************************
Add dependencies to my app
**************************


Add all dependencies required to run your app to a `requirements.txt` file in your app's directory. Read :ref:`build_config` for more details.



----


********
Name app
********

Simply use the ``--name`` flag when running your app, for example:

.. code:: bash

    lightning_app run app app.py --cloud --name my-awesome-app

Alternatively, you can change the name of the app in the ``.lightning`` file:

.. code:: bash

    ~/project/home ❯ cat .lightning
    name: my-awesome-app

The ``.lightning`` file is a general configuration file.
To learn more about optional configuration file parameters, see :class:`~lightning.app.utilities.packaging.app_config.AppConfig`.

------

********************
Choose Cloud Compute
********************

You can configure the hardware your app is running on by setting a :class:`~lightning.app.utilities.packaging.cloud_compute.CloudCompute` object onto the ``cloud_compute`` property of your work's.

Learn more with the :ref:`cloud_compute` guide
