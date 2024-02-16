:orphan:

######################################################
2. Develop a CLI with server and client code execution
######################################################

We've learned how to create a simple command-line interface. But in real-world use-cases, an App Builder wants to provide more complex functionalities where trusted code is executed on the client side.

Lightning provides a flexible way to create complex CLI without much effort.

In this example, we’ll create a CLI to dynamically run Notebooks:


----

**************************
1. Implement a complex CLI
**************************

First of all, lets' create the following file structure:

.. code-block:: python

    app_folder/
        commands/
            notebook/
                run.py
        app.py

We'll use the `Jupyter-Component <https://github.com/Lightning-AI/LAI-Jupyter-Component>`_. Follow the installation steps on the repo to install the Component.

Add the following code to ``commands/notebook/run.py``:

.. literalinclude:: commands/notebook/run.py

Add the following code to ``app.py``:

.. literalinclude:: app.py

----

**********************************************
2. Run the App and check the API documentation
**********************************************

In a terminal, run the following command and open ``http://127.0.0.1:7501/docs`` in a browser.

.. code-block:: python

    lightning_app run app app.py
    Your Lightning App is starting. This won't take long.
    INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view

----

***************************
3. Connect to a running App
***************************

In another terminal, connect to the running App.
When you connect to an App, the Lightning CLI is replaced by the App CLI. To exit the App CLI, you need to run ``lightning_app disconnect``.

.. code-block::

    lightning_app connect localhost

    Storing `run_notebook` under /Users/thomas/.lightning/lightning_connection/commands/run_notebook.py
    You can review all the downloaded commands under /Users/thomas/.lightning/lightning_connection/commands folder.
    You are connected to the local Lightning App.

To see a list of available commands:

.. code-block::

    lightning_app --help

    You are connected to the cloud Lightning App: localhost.
    Usage: lightning_app [OPTIONS] COMMAND [ARGS]...

    --help     Show this message and exit.

    Lightning App Commands
        run notebook Run a Notebook.


To find the arguments of the commands:

.. code-block::

    lightning_app run notebook --help

    You are connected to the cloud Lightning App: localhost.
    usage: notebook [-h] [--name NAME] [--cloud_compute CLOUD_COMPUTE]

    Run Notebook Parser

    optional arguments:
        -h, --help            show this help message and exit
        --name NAME
        --cloud_compute CLOUD_COMPUTE

----

********************
4. Execute a command
********************

And then you can trigger the command-line exposed by your App.

Run the first Notebook with the following command:

.. code-block:: python

    lightning_app run notebook --name="my_notebook"
    WARNING: Lightning Command Line Interface is an experimental feature and unannounced changes are likely.
    The notebook my_notebook was created.

And run a second notebook.

.. code-block:: python

    lightning_app run notebook --name="my_notebook_2"
    WARNING: Lightning Command Line Interface is an experimental feature and unannounced changes are likely.
    The notebook my_notebook_2 was created.

Here is a recording of the Lightning App:

.. video:: https://pl-public-data.s3.amazonaws.com/assets_lightning/commands_1.mp4
    :poster: https://pl-public-data.s3.amazonaws.com/assets_lightning/commands_1.png
    :width: 600
    :class: background-video
    :autoplay:
    :loop:
    :muted:

**************************
5. Disconnect from the App
**************************

To exit the App CLI, you need to run **lightning disconnect**.

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
   :header: 1. Develop a CLI with server side code only
   :description: Learn how to develop a simple CLI for your App.
   :col_css: col-md-6
   :button_link: cli.html
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
