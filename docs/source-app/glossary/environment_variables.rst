.. _environment_variables:

*********************
Environment Variables
*********************

If your app is using secrets or values you don't want to expose in your app code such as API keys or access tokens, you can use environment variables.

Lightning allows you to set environment variables when running the app from the CLI with the `lightning run app` command. You can use environment variables to pass any value such as API keys or other similar configurations to the app, avoiding having to stick them in the source code.

Set one or multiple variables using the **--env** option:

.. code:: bash

    lightning run app app.py --cloud --env FOO=BAR --env BAZ=FAZ

The environment variables are available in all flows and works, and can be accessed as follows:

.. code:: python

    import os

    print(os.environ["FOO"])  # BAR
    print(os.environ["BAZ"])  # FAZ

.. note::
	Environment variables are currently not encrypted.
