.. _environment_variables:

*********************
Environment Variables
*********************

If your App is using configuration values you don't want to commit with your App source code, you can use environment variables.

Lightning allows you to set environment variables when running the App from the CLI with the `lightning_app run app` command. You can use environment variables to pass any values to the App, and avoiding sticking those values in the source code.

Set one or multiple variables using the **--env** option:

.. code:: bash

    lightning_app run app app.py --cloud --env FOO=BAR --env BAZ=FAZ

Environment variables are available in all Flows and Works, and can be accessed as follows:

.. code:: python

    import os

    print(os.environ["FOO"])  # BAR
    print(os.environ["BAZ"])  # FAZ

.. note::
    Environment variables are not encrypted. For sensitive values, we recommend using :ref:`Encrypted Secrets <secrets>`.
