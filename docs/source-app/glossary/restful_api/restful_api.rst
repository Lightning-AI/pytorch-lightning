:orphan:

###########
RESTful API
###########

**Audience:** Users looking to create an API in their App to allow users to activate functionalities from external sources.

----

**********************
What is a RESTful API?
**********************

A RESTful API is a set of external URL routes exposed by a server that enables clients to trigger some functionalities, such as getting or putting some data, uploading files, etc..

This provides great flexibility for users as they can easily discover functionalities made available by the App Builders.

The Lightning App framework supports the four primary HTTP methods: `GET`, `POST`, `PUT`, `DELETE`.

These methods are guidelines to organize your RESTful Services and help users understand your functionalities.

* **`GET`:** Reads data from the server.
* **`POST`:** Creates new resources.
* **`PUT`:** Updates/replaces existing resources.
* **`DELETE`:** Deletes resources.

Learn more about `HTTP Methods for RESTful Services here <https://www.restapitutorial.com/introduction/whatisrest>`_.

The Lightning App framework uses the popular `FastAPI <https://fastapi.tiangolo.com/>`_ and `Pydantic <https://pydantic-docs.helpmanual.io/>`_ frameworks under the hood. This means you can use all their features while building your App.

----

**********
Learn more
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Develop a RESTful API
   :description: Learn how to develop an API for your App.
   :col_css: col-md-6
   :button_link: ../../workflows/build_rest_api/index_content.html
   :height: 150

.. raw:: html

        </div>
    </div>
