.. _secrets:

#################
Encrypted Secrets
#################

Is your App using data or values (for example: API keys or access credentials) that you don't want to expose in your App code? If the answer is yes, you'll want to use Secrets. Secrets are encrypted values that are stored in the Lightning.ai database and are decrypted at runtime.

.. tip::
	For non-sensitive configuration values, we recommend using :ref:`plain-text Environment Variables <environment_variables>`.

***************
What did we do?
***************

When a Lightning App (App) **runs in the cloud**, a Secret can be exposed to the App using environment variables.
The value of the Secret is encrypted in the Lightning.ai database, and is only decrypted and accessible to
LightningFlow (Flow) or LightningWork (Work) processes in the cloud (when you use the ``--cloud`` option running your App).

----

**********************
What were we thinking?
**********************

Many Apps require access to private data like API keys, access tokens, database passwords, or other credentials. You need to protect this data.
We developed this feature to provide you with a secure way to store this data in a way that is accessible to Apps so that they can authenticate third-party services/solutions.

----

*********************
Use Encrypted Secrets
*********************

To use Encrypted Secrets:

#. Log in to your lightning.ai account, go to **Secrets**, and create the Secret (provide a name and value for the secret).

    .. note:: Once you create a Secret, you can bind it to any of your Apps. You do not need to create a new Secret for each App if the Secret value is the same.

#. Prepare an environment variable to use with the Secret in your App.

#. Use the following command to add the Secret to your App:

.. code:: bash

    lightning run app app.py --cloud --secret <environment-variable>=<secret-name>

The environment variables are available in all Flows and Works, and can be accessed as follows:

.. code:: python

    import os

    print(os.environ["<environment-variable>"])

The ``--secret`` option can be used for multiple Secrets, and alongside the ``--env`` option.

Here's an example:

.. code:: bash

    lightning run app app.py --cloud --env FOO=bar --secret MY_APP_SECRET=my-secret --secret ANOTHER_SECRET=another-secret

----

Example
^^^^^^^

The best way to show you how to use Encrypted Secrets is with an example.

First, log in to your `lightning.ai account <https://lightning.ai/>`_ and create a Secret.

.. raw:: html

    <br />
    <video id="background-video" autoplay loop muted controls poster="https://pl-flash-data.s3.amazonaws.com/assets_lightning/docs/images/storage/encrypted_secrets_login.png" width="100%">
        <source src="https://pl-flash-data.s3.amazonaws.com/assets_lightning/docs/images/storage/encrypted_secrets_login.mp4" type="video/mp4" width="100%">
    </video>
    <br />
    <br />

.. note::
    Secret names must start with a letter and can only contain letters, numbers, dashes, and periods. The Secret names must comply with `RFC1123 naming conventions <https://www.rfc-editor.org/rfc/rfc1123>`_. The Secret value has no restrictions.

After creating a Secret named ``my-secret`` with the value ``some-secret-value`` we'll bind it to the environment variable ``MY_APP_SECRET`` within our App. The binding is accomplished by using the ``--secret`` option when running the App from the Lightning CLI.

The ``--secret``` option works similar to ``--env``, but instead of providing a value, you provide the name of the Secret that is replaced with with the value that you want to bind to the environment variable:

.. code:: bash

    lightning run app app.py --cloud --secret MY_APP_SECRET=my-secret

The environment variables are available in all Flows and Works, and can be accessed as follows:

.. code:: python

    import os

    print(os.environ["MY_APP_SECRET"])

This code prints out ``some-secret-value``.
