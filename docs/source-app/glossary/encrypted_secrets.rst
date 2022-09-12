.. _encrypted_secrets:

#################
Encrypted Secrets
#################

Private data (API keys, database passwords, or other credentials), required by Lightning Apps, can now be encrypted using the ``--secret`` flag.

----

***************
What did we do?
***************

The ``MY_SECRET`` environment variable has been added and the flag ``--secret`` has been added to the ``lightning run app`` command.

When a Lightning App (App) **runs in the cloud**, the ``MY_SECRET`` environment variable is populated with the value of the
referenced secret. The value of the secret is encrypted in the database, and is only decrypted and accessible to
LightningFlow or LightningWork processes in the cloud.

----

**********************
What were we thinking?
**********************

We understand that many Lightning Apps require access to private data like API keys, database passwords, or other credentials.
We developed this feature because we know that you need a secure way to store this data in a way that is accessible to Apps so that they can authenticate third party services/solutions.

----

****************
Encrypt a secret
****************

.. note:: Secrets can only be used for Apps running in cloud.

To encrypt your secret:

.. code:: bash

    lightning run app --cloud --secret MY_SECRET=<secret-name> <file with the secret>

Here's an example:

.. code:: bash

    lightning run app --cloud --secret MY_SECRET=my-super-secret-name app.py
