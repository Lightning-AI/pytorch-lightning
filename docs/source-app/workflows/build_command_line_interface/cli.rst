:orphan:

###########################################
1. Develop a CLI with server side code only
###########################################

We are going to learn how to create a simple command-line interface.

Lightning provides a flexible way to create complex CLI without much effort.

----

*************************
1. Implement a simple CLI
*************************

To create your first CLI, you need to override the :class:`~lightning.app.core.flow.LightningFlow.configure_commands` hook and return a list of dictionaries where the keys are the commands and the values are the server side handlers.

First, create a file ``app.py`` and copy-paste the following code in to the file:

.. literalinclude:: example_command.py

----

**************
2. Run the App
**************

Execute the following command in a terminal:

.. code-block::

    lightning_app run app app.py

The following appears the terminal:

.. code-block::

    Your Lightning App is starting. This won't take long.
    INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
    []

----

***************************
3. Connect to a running App
***************************

In another terminal, connect to the running App.
When you connect to an App, the Lightning CLI is replaced by the App CLI. To exit the App CLI, you need to run ``lightning_app disconnect``.

.. code-block::

    lightning_app connect localhost

To see a list of available commands:

.. code-block::

    lightning_app --help
    You are connected to the cloud Lightning App: localhost.
    Usage: lightning_app [OPTIONS] COMMAND [ARGS]...

    --help     Show this message and exit.

    Lightning App Commands
        add Add a name.

To find the arguments of the commands:

.. code-block::

    lightning_app add --help
    You are connected to the cloud Lightning App: localhost.
    Usage: lightning_app add [ARGS]...

    Options
        name: Add description

----

********************
4. Execute a command
********************

Trigger the command line exposed by your App:

.. code-block::

    lightning_app add --name=my_name
    WARNING: Lightning Command Line Interface is an experimental feature and unannounced changes are likely.

In your first terminal, **Received name: my_name** and **["my_name"]** are printed.

.. code-block::

    Your Lightning App is starting. This won't take long.
    INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
    []
    Received name: my_name
    ["my_name]

----

**************************
5. Disconnect from the App
**************************

To exit the App CLI, you need to run ``lightning_app disconnect``.

.. code-block::

    lightning_app disconnect
    You are disconnected from the local Lightning App.

----

**********
Learn more
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: 2. Implement a CLI with client side code execution
   :description: Learn how to develop a complex API for your application
   :col_css: col-md-6
   :button_link: cli_client.html
   :height: 150

.. displayitem::
   :header: Develop a RESTful API
   :description: Learn how to develop an API for your application.
   :col_css: col-md-6
   :button_link: ../build_rest_api/index.html
   :height: 150

.. raw:: html

        </div>
    </div>
