.. _encrypt_secrets:

########################
Encrypt your credentials
########################

Is your App using data or values (for example: API keys or access credentials) that you don't want to expose in your App code? If the answer is yes, you'll want to use Secrets. Secrets are encrypted values that are stored in the Lightning.ai database and are decrypted at runtime.

.. tip::
	For non-sensitive configuration values, we recommend using :ref:`plain-text Environment Variables <environment_variables>`.

----

.. include:: ../../workflows/encrypted_secrets/encrypt_secrets_content.rst
