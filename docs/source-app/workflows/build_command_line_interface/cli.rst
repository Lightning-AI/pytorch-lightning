:orphan:

**************************************
Develop a CLI without client side code
**************************************

1. Implement a simple CLI
^^^^^^^^^^^^^^^^^^^^^^^^^

In order to create your first CLI, you need to override the :class:`~lightning_app.core.flow.LightningFlow.configure_commands` hook and return a list of dictionaries where the keys are the commands and the values are the server-side handlers.

.. literalinclude:: example_command.py

After copy-pasting the code above to a file ``app.py``, execute the following command in your terminal in your first terminal.

2. Run the App
^^^^^^^^^^^^^^

.. code-block::

    lightning run app app.py

And you can find the following in your terminal:

.. code-block::

    Your Lightning App is starting. This won't take long.
    INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
    []

3. Connect to a running App
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In another terminal, you connect to the running application.
When you connect to an application, Lightning CLI is replaced by the App CLI. To exit the App CLI, you need to run lightning disconnect.

.. code-block::

    lightning connect localhost

And list of the available commands:

.. code-block::

    lightning --help
    You are connected to the cloud Lightning App: localhost.
    Usage: lightning [OPTIONS] COMMAND [ARGS]...

    --help     Show this message and exit.

    Lightning App Commands
        add Description

And you can find the arguments of the commands.

.. code-block::

    lightning add --help
    You are connected to the cloud Lightning App: localhost.
    Usage: lightning add [ARGS]...

    Options
        name: Add description

4. Execute a command
^^^^^^^^^^^^^^^^^^^^

And then you can trigger the command line exposed by your application.

.. code-block::

    lightning add --name=my_name
    WARNING: Lightning Command Line Interface is an experimental feature and unannounced changes are likely.

In your first terminal, **Received name: my_name** and **["my_name"]** are printed.

.. code-block::

    Your Lightning App is starting. This won't take long.
    INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
    []
    Received name: my_name
    ["my_name]

5. Disconnect
^^^^^^^^^^^^^

.. code-block::

    lightning disconnect
    You are disconnected of the local Lightning App.

----

**********
Learn more
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Develop a CLI with server and client code execution
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
