*********
Objective
*********

Create a simple application where users can upload files and list the uploaded files.

----

*****************
Final Application
*****************

Here is a recording of the final application built in this example tested with pytest.

.. raw:: html

   <iframe width="100%" height="290" src="https://pl-flash-data.s3.amazonaws.com/assets_lightning/file_server.mp4" frameborder="0" allowfullscreen></iframe>

----

*************
System Design
*************

In order to create such application, we need to build two components and an application:

* A **File Server Component** that gives you the ability to download or list files shared with your application. This is particularly useful when you want to trigger an ML job but your users need to provide their own data or if the user wants to download the trained checkpoints.

* A **Test File Server** Component to interact with the file server.

* An application putting everything together and its associated pytest tests.

----

********
Tutorial
********

Let's dive in on how to create such application and component:

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: 1. Implement the File Server general structure
   :description: Put together the shape of the component
   :col_css: col-md-4
   :button_link: file_server_step_1.html
   :height: 180
   :tag: Basic

.. displayitem::
   :header: 2. Implement the File Server upload and list files methods
   :description: Add the core functionalities to the component
   :col_css: col-md-4
   :button_link: file_server_step_2.html
   :height: 180
   :tag: Basic

.. displayitem::
   :header: 3. Implement a File Server Testing Component
   :description: Create a component to test the file server
   :col_css: col-md-4
   :button_link: file_server_step_3.html
   :height: 180
   :tag: Intermediate


.. displayitem::
   :header: 4. Implement tests for the File Server component with pytest
   :description: Create an app to validate the upload and list files endpoints
   :col_css: col-md-4
   :button_link: file_server_step_4.html
   :height: 180
   :tag: Intermediate

.. raw:: html

        </div>
    </div>
