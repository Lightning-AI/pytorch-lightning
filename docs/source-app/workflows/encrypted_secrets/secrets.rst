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

.. include:: ../../workflows/encrypted_secrets/encrypt_secrets_content.rst
