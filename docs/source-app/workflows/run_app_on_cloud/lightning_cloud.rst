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

    lightning run app app.py --cloud

.. note::
    Running on Lightning Cloud requires signing up to lightning.ai and Lightning Credits to use resources such as compute and storage. By default every user has 3 free credits every month, if your app requires more credits you will need to add more credits to your account.  Read :ref:`credits` to learn more.

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

    lightning run app app.py --cloud --name my-awesome-app

Alternatively, you can change the name of the app in the ``.lightning`` file:

.. code:: bash

    ~/project/home ❯ cat .lightning
    name: my-awesome-app

The ``.lightning`` file is a general configuration file.
To learn more about optional configuration file parameters, see :class:`~lightning.utilities.packaging.app_config.AppConfig`.

------

********************
Choose Cloud Compute
********************

You can configure the hardware your app is running on by setting a :class:`~lightning.utilities.packaging.cloud_compute.CloudCompute` object onto the ``cloud_compute`` property of your work's.

Learn more with the :ref:`cloud_compute` guide.


