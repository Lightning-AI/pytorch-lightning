.. _secrets:

#################
Encrypted Secrets
#################

Encrypted Secrets allow you to pass private data to your apps, like API keys, access tokens, database passwords, or other credentials, in a secure way without exposing them in your code.
Secrets provide you with a secure way to store this data in a way that is accessible to Apps so that they can authenticate third-party services/solutions.

.. tip::
    For non-sensitive configuration values, we recommend using :ref:`plain-text Environment Variables <environment_variables>`.

************
Add a secret
************

Add the secret to your profile on lightning.ai.
Log in to your lightning.ai account > **Profile** > **Secrets** tab > click the **+New** button.
Provide a name and value to your secret, for example, name could be "github_api_token".

.. note::
    Secret names must start with a letter and can only contain letters, numbers, dashes, and periods. The Secret names must comply with `RFC1123 naming conventions <https://www.rfc-editor.org/rfc/rfc1123>`_. The Secret value has no restrictions.

.. video:: https://pl-public-data.s3.amazonaws.com/assets_lightning//encrypted_secrets_login.mp4
    :poster: https://pl-public-data.s3.amazonaws.com/assets_lightning//encrypted_secrets_login.png
    :width: 600
    :class: background-video
    :autoplay:
    :loop:
    :muted:

************
Use a secret
************

1. Add an environment variable to your app to read the secret. For example, add an "api_token" environment variable:

.. code:: python

    import os

    component.connect(api_token=os.environ["api_token"])

2. Pass the secret to your app run with the following command:

.. code:: bash

    lightning_app run app app.py --cloud --secret <environment-variable>=<secret-name>

In this example, the command would be:

.. code:: bash

    lightning_app run app app.py --cloud --secret api_token=github_api_token


The ``--secret`` option can be used for multiple Secrets, and alongside the ``--env`` option.

Here's an example:

.. code:: bash

    lightning_app run app app.py --cloud --env FOO=bar --secret MY_APP_SECRET=my-secret --secret ANOTHER_SECRET=another-secret


----

******************
How does this work
******************

When a Lightning App (App) **runs in the cloud**, a Secret can be exposed to the App using environment variables.
The value of the Secret is encrypted in the Lightning.ai database, and is only decrypted and accessible to
LightningFlow (Flow) or LightningWork (Work) processes in the cloud (when you use the ``--cloud`` option running your App).
