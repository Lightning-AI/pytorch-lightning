.. _secrets:

#################
Encrypted Secrets
#################

We understand that many Apps require access to private data like API keys, access tokens, database passwords, or other credentials. And that you need to protect this data.

Secrets provie a secure way to make private data like API keys or passwords accessible to your app, without hardcoding. You can use secrets to authenticate third-party services/solutions.

.. tip::
	For non-sensitive configuration values, we recommend using :ref:`plain-text Environment Variables <environment_variables>`.

*******************
Overview of Secrets
*******************

The ``--secret`` option has been added to the **lightning run app** command. ``--secret`` can be used by itself or alongside ``--env``.

When a Lightning App (App) **runs in the cloud**, the Secret can be exposed to the App using environment variables.
The value of the Secret is encrypted in the Lightning.ai database, and is only decrypted and accessible to
LightningFlow (Flow) or LightningWork (Work) processes in the cloud (when you use the ``--cloud`` option running your App).

----

*********************
Use Encrypted Secrets
*********************

First, a Secret must be created using the admin web UI. Once you create a Secret, you can bind it to any of your Apps. You do not need to create a new Secret for each App if the Secret value is the same.

.. note::
    Secret names must start with a letter and can only contain letters, numbers, dashes, and periods. The Secret names must comply with `RFC1123 naming conventions <https://www.rfc-editor.org/rfc/rfc1123>`_. The Secret value has no restrictions.

In the example below, we already used the admin UI to create a Secret named ``my-secret`` with the value ``some-value``` and will bind it to the environment variable ``MY_APP_SECRET`` within our App. The binding is accomplished by using the ``--secret`` option when running the App from the Lightning CLI.

The ``--secret``` option works similar to ``--env``, but instead of providing a value, you provide the name of the Secret which will be replaced with with the value that you want to bind to the environment variable:

.. code:: bash

    lightning run app app.py --cloud --secret MY_APP_SECRET=my-secret

The environment variables are available in all Flows and Works, and can be accessed as follows:

.. code:: python

    import os

    print(os.environ["MY_APP_SECRET"])

The code above will print out ``some-value``.

The ``--secret`` option can be used for multiple Secrets, and alongside the ``--env`` option:

.. code:: bash

    lightning run app app.py --cloud --env FOO=bar --secret MY_APP_SECRET=my-secret --secret ANOTHER_SECRET=another-secret
