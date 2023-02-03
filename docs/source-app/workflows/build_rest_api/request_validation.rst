:orphan:

***********************
Add Requests Validation
***********************

The Lightning App framework uses the popular `FastAPI <https://fastapi.tiangolo.com/>`_ and `Pydantic <https://pydantic-docs.helpmanual.io/>`_ frameworks under the hood. This means you can use all their features while building your App.

pydantic enables fast data validation and settings management using Python type annotations and FastAPI is a modern, fast (high-performance), web framework for building APIs.

You can easily use pydantic by defining your own payload format.

.. literalinclude:: models.py

Then, type your handler input with your custom model.

.. literalinclude:: post_example_pydantic.py

After running the updated App, the App documentation ``/name`` has changed and takes JSON with ``{"name": ...}`` as input.

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/rest_post_pydantic.png
   :alt: Rest API with pydantic
   :width: 100 %

You can invoke the RESTful API route ``/name`` with the following command:

.. code-block:: bash

    curl -X 'POST' \
    'http://127.0.0.1:7501/name' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "name": "my_name"
    }'

.. note::

    Using curl, you can pass a JSON payload using the ``-d`` argument.

----

**********
Learn more
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Add an API Route to your App
   :description: Learn how to develop a simple API for your App.
   :col_css: col-md-6
   :button_link: add_api.html
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
