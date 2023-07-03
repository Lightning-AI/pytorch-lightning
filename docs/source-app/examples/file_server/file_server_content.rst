

*********
Our Goal
*********

Create a simple Lightning App (App) that allows users to upload files and list the uploaded files.

----

*************
Completed App
*************

Here is a recording of the final App built in this example, tested with pytest.

.. video:: https://pl-public-data.s3.amazonaws.com/assets_lightning/file_server.mp4
    :poster: https://pl-public-data.s3.amazonaws.com/assets_lightning/file_server.png
    :width: 600
    :class: background-video
    :autoplay:
    :loop:
    :muted:

----

**********
App Design
**********

In order to create this App, we need to develop two components and an App:

* A **File Server Component** that gives you the ability to download or list files shared with your App. This is particularly useful when you want to trigger an ML job but your users need to provide their own data or if the user wants to download the trained checkpoints.

* A **Test File Server** Component to interact with the file server.

* An App putting everything together and the App's associated pytest tests.

----

********
Tutorial
********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Step 1: Implement the File Server general structure
   :description: Put together the shape of the Component
   :col_css: col-md-4
   :button_link: file_server_step_1.html
   :height: 180
   :tag: Basic

.. displayitem::
   :header: Step 2: Implement the File Server upload and list files methods
   :description: Add the core functionalities to the Component
   :col_css: col-md-4
   :button_link: file_server_step_2.html
   :height: 180
   :tag: Basic

.. displayitem::
   :header: Step 3: Implement a File Server Testing Component
   :description: Create a Component to test the file server
   :col_css: col-md-4
   :button_link: file_server_step_3.html
   :height: 180
   :tag: Intermediate

.. displayitem::
   :header: Step 4: Implement tests for the File Server component with pytest
   :description: Create an App to validate the upload and list files endpoints
   :col_css: col-md-4
   :button_link: file_server_step_4.html
   :height: 180
   :tag: Intermediate

.. raw:: html

        </div>
    </div>
