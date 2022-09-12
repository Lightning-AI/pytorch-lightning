#############################
Level 3: Run app on the cloud
#############################
**Audience:** Users who want to run an app on the Lightning Cloud.

**Prereqs:** You have an app already running locally.

----

****************************
What is the Lightning Cloud?
****************************
The Lightning Cloud is the platform that we've created to interface with the cloud providers. Today
the Lightning Cloud supports AWS.

.. note:: Support for GCP and Azure is coming soon!

To use the Lightning Cloud, you buy credits that are used to pay the cloud providers. If you want to run
on your own AWS credentials, please contact us (support@lightning.ai) so we can get your clusters set up for you.

----

****************
Run on the cloud
****************
To run the app on the cloud, simply add **--cloud** to the command

.. code:: bash

    lightning run app app.py --cloud

Lightning packages everything in that folder and uploads it to the cloud. Your code will be visible to everyone (just like Github).

.. note::  To have private Lightning Apps, you'll need to upgrade your account. To upgrade, contact us `<support@lightning.ai>`_.

----

************
Ignore files
************
Lightning sends everything in your Lightning App folder to the cloud. If you want to ignore certain files (such as datasets),
use the **.lightningignore** file which works just like the **.gitignore**

.. code:: bash

    touch .lightningignore

----

*******************
Manage requirements
*******************
A Lightning App is simply a Python file. However, when running on the cloud, it is encouraged that you add
a **requirements.txt** file so that the platform knows what requirements your Lightning App needs.

If you require custom Docker images, each LightningWork has the ability to have a private Docker image.
