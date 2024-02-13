:orphan:

############################
Add an API Route to your App
############################

In order to add a new route, you need to override the :class:`~lightning.app.core.flow.LightningFlow.configure_api` hook and return a list of :class:`~lightning.app.api.http_methods.HttpMethod` such as :class:`~lightning.app.api.http_methods.Get`, :class:`~lightning.app.api.http_methods.Post`, :class:`~lightning.app.api.http_methods.Put`, :class:`~lightning.app.api.http_methods.Delete`.

----

**********************
1. Create a simple App
**********************

We're going to create a single route ``/name`` that takes a string input ``name`` and stores the value within the ``names`` attribute of the flow state.

Create a file called ``app.py`` and copy-paste the following code in to the file:

.. literalinclude:: post_example.py

----

**************
2. Run the App
**************

Execute the following command in a terminal:

.. code-block::

     lightning_app run app app.py

The following appears:

.. code-block::

     Your Lightning App is starting. This won't take long.
     INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view

----

****************
3. Check the API
****************

The Lightning App framework automatically generates API documentation from your App using `Swagger UI <https://fastapi.tiangolo.com/features/#automatic-docs>`_.

You can access it by accessing the following URL: ``http://127.0.0.1:7501/docs`` in your browser and validate your API with the route ``/name`` directly from the documentation page as shown below.

.. video:: https://pl-public-data.s3.amazonaws.com/assets_lightning/rest_post.mp4
    :poster: https://pl-public-data.s3.amazonaws.com/assets_lightning/rest_png.png
    :width: 600
    :class: background-video
    :autoplay:
    :loop:
    :muted:

Alternatively, you can invoke the route directly from a second terminal using `curl <https://curl.se/>`_.

.. code-block::

     curl -X 'POST' \
     'http://127.0.0.1:7501/name?name=my_name' \
     -H 'accept: application/json' \
     -d ''

     "The name my_name was registered"

And you can see the following in your first terminal running your App.

.. code-block::

     Your Lightning App is starting. This won't take long.
     INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
     []
     ["my_name"]

**************************************
Develop a command line interface (CLI)
**************************************

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Add Requests Validation
   :description: Learn how to use pydantic with your API.
   :col_css: col-md-6
   :button_link: request_validation.html
   :height: 150

.. displayitem::
   :header: Develop a Command Line Interface (CLI)
   :description: Learn how to develop an CLI for your App.
   :col_css: col-md-6
   :button_link: ../build_command_line_interface/index.html
   :height: 150

.. raw:: html

        </div>
    </div>
