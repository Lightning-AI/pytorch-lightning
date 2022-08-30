:orphan:

***********************************
Develop a CLI with client side code
***********************************


In the previous section, we learned how to create a simple command line interface. In more realistic use-cases, an app builder wants to provide more complex functionalities where trusted code is executed on the client side.

Lightning provides a flexible way to create complex CLI without effort.

1. Implement a complex CLI
^^^^^^^^^^^^^^^^^^^^^^^^^^

In the example below, we create a CLI to dynamically run notebooks with the following structures.

.. code-block:: python

    app_folder/
        commands/
            notebook/
                run.py
        app.py

Furthermore, we are using the `Jupyter-Component <https://github.com/Lightning-AI/LAI-Jupyter-Component>`_. Follow the installation steps on the repo.

In the ``commands/notebook/run.py``, add the following code:

.. literalinclude:: commands/notebook/run.py

And in the ``app.py``, add the following code:

.. literalinclude:: app.py


2. Run the App and check the API documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In your first terminal, run the following command and open the ``http://127.0.0.1:7501/view`` in browser.

.. code-block:: python

    lightning run app app.py
    Your Lightning App is starting. This won't take long.
    INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view

3. Connect to a running App
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In another terminal, you connect to the running application.
When you connect to an application, Lightning CLI is replaced by the App CLI. To exit the App CLI, you need to run lightning disconnect.

.. code-block::

    lightning connect localhost

    Storing `run_notebook` under /Users/thomas/.lightning/lightning_connection/commands/run_notebook.py
    You can review all the downloaded commands under /Users/thomas/.lightning/lightning_connection/commands folder.
    You are connected to the local Lightning App.

And list of the available commands:

.. code-block::

    lightning --help

    You are connected to the cloud Lightning App: localhost.
    Usage: lightning [OPTIONS] COMMAND [ARGS]...

    --help     Show this message and exit.

    Lightning App Commands
        run notebook Description


And you can find the arguments of the commands.

.. code-block::

    lightning run notebook --help

    You are connected to the cloud Lightning App: localhost.
    usage: notebook [-h] [--name NAME] [--cloud_compute CLOUD_COMPUTE]

    Run Notebook Parser

    optional arguments:
        -h, --help            show this help message and exit
        --name NAME
        --cloud_compute CLOUD_COMPUTE

4. Execute a command
^^^^^^^^^^^^^^^^^^^^

And then you can trigger the command line exposed by your application.

Run a first notebook with the following command:

.. code-block:: python

    lightning run notebook --name="my_notebook"
    WARNING: Lightning Command Line Interface is an experimental feature and unannounced changes are likely.
    The notebook my_notebook was created.

And run a second notebook by changing its name as follows:

.. code-block:: python

    lightning run notebook --name="my_notebook_2"
    WARNING: Lightning Command Line Interface is an experimental feature and unannounced changes are likely.
    The notebook my_notebook_2 was created.

Here is a recording of the Lightning App described above.

.. raw:: html

    <br />
    <video id="background-video" autoplay loop muted controls poster="https://pl-flash-data.s3.amazonaws.com/assets_lightning/commands_1.png" width="100%">
        <source src="https://pl-flash-data.s3.amazonaws.com/assets_lightning/commands_1.mp4" type="video/mp4" width="100%">
    </video>
    <br />
    <br />

5. Disconnect from the App
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    lightning disconnect
    You are disconnected from the local Lightning App.

----

**********
Learn more
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Develop a CLI with server side code only
   :description: Learn how to develop a simple API for your application
   :col_css: col-md-6
   :button_link: cli.html
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
