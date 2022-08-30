:orphan:

***********************************
Develop a CLI with client side code
***********************************

In the previous section, we learned how to create a simple command line interface. In more realistic use-cases, an app builder wants to provide more complex functionalities where trusted code is executed on the client side.

Lightning provides a flexible way to create complex CLI without effort.

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

In your first terminal, run the following command and open the ``http://127.0.0.1:7501/view`` in browser.

.. code-block:: python

    lightning run app app.py
    Your Lightning App is starting. This won't take long.
    INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view

And a second terminal, run a first notebook

.. code-block:: python

    lightning run-notebook --name="my_notebook"
    WARNING: Lightning Command Line Interface is an experimental feature and unannounced changes are likely.
    The notebook my_notebook was created.

And run a second notebook.

.. code-block:: python

    lightning run-notebook --name="my_notebook_2"
    WARNING: Lightning Command Line Interface is an experimental feature and unannounced changes are likely.
    The notebook my_notebook_2 was created.

Here is a recording of the Lightning App described above.

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
