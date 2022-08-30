:orphan:

***********************************
Develop a CLI with client side code
***********************************

We've learned how to create a simple command-line interface. But in real-world use-cases, an App Builder wants to provide more complex functionalities where trusted code is executed on the client side.

Lightning provides a flexible way to create complex CLI without much effort.

Here's an example:

We're going to create a CLI to dynamically run Notebooks using the following:

.. code-block:: python

    app_folder/
        commands/
            notebook/
                run.py
        app.py

We're also going to use the `Lightning Jupyter-Component <https://github.com/Lightning-AI/LAI-Jupyter-Component>`_. Follow the installation steps on the repo to install the Component.

#. In the ``commands/notebook/run.py`` file, add the following code:

     .. literalinclude:: commands/notebook/run.py

#.   In the ``app.py`` file, add the following code:

     .. literalinclude:: app.py

#. Open a terminal and run the following command:

     .. code-block:: python

         lightning run app app.py
         Your Lightning App is starting. This won't take long.
         INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view

#. Open this link ``http://127.0.0.1:7501/view`` in a web browser.

#. Open a second terminal, and run this notebook:

     .. code-block:: python

         lightning run-notebook --name="my_notebook"
         WARNING: Lightning Command Line Interface is an experimental feature and unannounced changes are likely.
         The notebook my_notebook was created.

#. Now run a second notebook:

     .. code-block:: python

         lightning run-notebook --name="my_notebook_2"
         WARNING: Lightning Command Line Interface is an experimental feature and unannounced changes are likely.
         The notebook my_notebook_2 was created.

This is what the App is going to look like:

.. raw:: html

    <br />
    <video id="background-video" autoplay loop muted controls poster="https://pl-flash-data.s3.amazonaws.com/assets_lightning/commands.png" width="100%">
        <source src="https://pl-flash-data.s3.amazonaws.com/assets_lightning/commands.mp4" type="video/mp4" width="100%">
    </video>
    <br />
    <br />

----

**********
Learn more
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Develop a CLI with server side code only.
   :description: Learn how to develop a simple CLI for your App.
   :col_css: col-md-6
   :button_link: cli.html
   :height: 150

.. displayitem::
   :header: Develop a RESTful API.
   :description: Learn how to develop an API for your App.
   :col_css: col-md-6
   :button_link: ../build_rest_api/index.html
   :height: 150

.. raw:: html

        </div>
    </div>
