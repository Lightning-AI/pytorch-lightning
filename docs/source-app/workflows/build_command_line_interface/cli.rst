:orphan:

**************************************
Develop a CLI without client side code
**************************************

In order to create your first CLI, you need to override the :class:`~lightning_app.core.flow.LightningFlow.configure_commands` hook and return a list of dictionaries where the keys are the commands and the values are the server-side handlers.

.. literalinclude:: example_command.py

After copy-pasting the code above to a file ``app.py``, execute the following command in your terminal in your first terminal.

.. code-block::

    lightning run app app.py

And you find the following:

.. code-block::

    Your Lightning App is starting. This won't take long.
    INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
    []

In another terminal, you can trigger the command line exposed by your application.

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
