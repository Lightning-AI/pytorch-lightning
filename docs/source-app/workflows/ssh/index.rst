
#################################
Debug cloud apps via SSH
#################################

**Audience:** Users that want to debug their cloud apps on their own machine

----

**********************************************************
Add SSH key to lightning
**********************************************************
Open a terminal and run the following command:

.. code:: bash

   $ ssh-keygen -t ed25519 -C "your_email@example.com"

This creates a new SSH key, using the provided email as a label.

At the prompt, type a secure passphrase.

.. code:: bash

   > Enter passphrase (empty for no passphrase): [Type a passphrase]
   > Enter same passphrase again: [Type passphrase again]

Next, start your ssh-agent in the background:

.. code:: bash

   $ eval "$(ssh-agent -s)"
   > Agent pid 12345

Add your generated ssh key:

.. code:: bash

   $ ssh-add ~/.ssh/id_ed25519

Lastly, verify your ssh-key is properly loaded:

.. code:: bash

   $ ssh-add -L
   ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAn8mYRnRG1banQcfXPCUC6R8FvQS+YgfIsl70/dD3Te your_email@example.com

----

**********************************************************
Managing SSH keys with lightning CLI
**********************************************************
Open a terminal and run the following command:

.. code:: bash

   $ lightning add ssh-key --public-key ~/.ssh/id_ed25519.pub

Next, list your configured SSH keys:

.. code:: bash

   $ lightning list ssh-keys

You are now ready to access your Lightning Flow and Work containers.

----

**********************************************************
Accessing your Lightning Flow and Work containers via SSH
**********************************************************

Ensure you have a running Lightning application in the cloud:

.. code:: bash

   $ lightning run app app.py --cloud --name my-app

Now, use the CLI to access any of the available components:

.. code:: bash

   $ lightning ssh
