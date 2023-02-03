
########################
Debug cloud apps via SSH
########################

**Audience:** Users that want to debug their cloud apps on their own machine.

SSH allows you to access Lightning components running on Lightning cloud in real-time using your own local machine.
This can be used for debugging apps running on the cloud: inspecting file system or monitoring the runtime environment.
This is often the case with applications that rely on GPUs.

You can inspect the entire file system and monitor the runtime environment.

----

************************
Add SSH key to Lightning
************************

Before you can SSH to cloud machines, you will need to generate a new private SSH key, add it to the SSH agent, and add the public SSH key to your account on Lightning.
The following steps are one-time setup to allow you to SSH into your cloud machines.


Step 1: Create an SSH key
==========================

Open a terminal and run the following command (replace email with the address you used in your lightning.ai account):

.. code:: bash

   # make the ssh key (if you don't have one)
   $ ssh-keygen -t ed25519 -C "your_email@example.com"

This creates a new SSH key, using the provided email as a label.

At the prompt, type a secure passphrase.

.. code:: bash

   > Enter passphrase (empty for no passphrase): [Type a passphrase]
   > Enter same passphrase again: [Type passphrase again]


Step 2: Add your key to Lightning
=================================

You can add SSH keys using Lightning.ai website (Lightning.ai > Profile > Keys) or via this CLI command:

.. code:: bash

   $ lightning create ssh-key --public-key ~/.ssh/id_ed25519.pub

You are now ready to access your Lightning Flow and Work containers.

----

**********************************************************
SSH to your cloud app
**********************************************************

Ensure you have a running Lightning application in the cloud:

.. code:: bash

   $ lightning run app app.py --cloud --name my-app


Next, start your ssh-agent in the background:

.. code:: bash

   # add the key to the ssh-agent (to avoid having to explicitly state key on each connection)
   # to start the agent, run the following
   $ eval "$(ssh-agent -s)"
   > Agent pid 12345

Add your generated ssh key:

.. code:: bash

   $ ssh-add ~/.ssh/id_ed25519

Verify your ssh-key is properly loaded:

.. code:: bash

   $ ssh-add -L
   ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAn8mYRnRG1banQcfXPCUC6R8FvQS+YgfIsl70/dD3Te your_email@example.com


You can now SSH any app you are running on the cloud.

To view all apps you can simple use this following:

.. code:: bash

   $ lightning list apps

Or, to select an app via a prompt:

.. code:: bash

   $ lightning ssh

To connect to a specific app flow use:

.. code:: bash

   $ lightning list apps
   $ lightning ssh --app-name <your-app-name> # taken from previous app listing

To connect to a LightningWork component use:

.. code:: bash

   $ lightning ssh --app-name <your-app-name> --component-name <work-name>

The component name is the variable name of your LightningWork instances in Python.
If you want to access your flow, use "flow" as component name.
