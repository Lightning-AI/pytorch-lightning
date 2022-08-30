:orphan:

###########
RESTful API
###########

**Audience:** Users looking to create an API in their application to let users activate functionalities from external sources.

----

***********************
What is a RESTful API ?
***********************

A RESTful API is a set of external url routes exposed by a server that enables clients to trigger some functionalities such as getting or putting some data, uploading files, etc..

This provides great flexibility for clients as they can easily discover functionalities made available by the App Builders.

The Lightning App framework supports the 4 traditional http methods.

They are guidelines to organize your RESTful Services and help clients understand your functionalities.

* **Get** is used to read data from the server.
* **Post** is used to create new resources.
* **Put** is used to update/replace existing resources.
* **Delete** is used to delete resources.

Learn more with `HTTP Methods for RESTful Services <https://www.restapitutorial.com/lessons/httpmethods.html#:~:text=The%20primary%20or%20most%2Dcommonly,but%20are%20utilized%20less%20frequently.>`_.

The Lightning App framework uses the popular `FastAPI <https://fastapi.tiangolo.com/>`_ and `Pydantic <https://pydantic-docs.helpmanual.io/>`_ frameworks under the hood e.g you can use all their features while building your application.

----

**********
Learn more
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Develop a RESTful API
   :description: Learn how to develop an API for your application.
   :col_css: col-md-6
   :button_link: ../../workflows/build_rest_api/index_content.html
   :height: 150

.. raw:: html

        </div>
    </div>
