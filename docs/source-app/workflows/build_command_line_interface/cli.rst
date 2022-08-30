:orphan:

**************************************
Develop a CLI without client side code
**************************************

In order to create your first CLI, you need to override the :class:`~lightning_app.core.flow.LightningFlow.configure_commands` hook and return a list of dictionaries where the keys are the commands and the values are the server side handlers.

Here's an example:

#. Create a file called ``app.py`` and copy-paste the following code in to the file:

     .. literalinclude:: example_command.py

#. Execute the following command in a terminal:

     .. code-block:: python

         lightning run app app.py

     The following appears:

     .. code-block:: python

     Your Lightning App is starting. This won't take long.
     INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
     []

#. In another terminal, trigger the command line exposed by your App:

     .. code-block:: python

         lightning add --name=my_name
         WARNING: Lightning Command-line Interface is an experimental feature and unannounced changes are likely.

#. In your first terminal, **Received name: my_name** and **["my_name"]** are printed.

     .. code-block:: python

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
   :description: Learn how to develop a complex API for your App.
   :col_css: col-md-6
   :button_link: cli_client.html
   :height: 150

.. displayitem::
   :header: Develop a RESTful API
   :description: Learn how to develop an API for your App.
   :col_css: col-md-6
   :button_link: ../build_rest_api/index.html
   :height: 150

.. raw:: html

        </div>
    </div>
