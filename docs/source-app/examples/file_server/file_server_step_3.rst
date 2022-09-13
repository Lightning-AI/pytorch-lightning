:orphan:

#################################################
Step 3: Implement a File Server Testing Component
#################################################

Let's dive in on how to implement a testing component for a server.

This component needs to test two things:

* The **/upload_file/** endpoint by creating a file and sending its content to it.

* The **/** endpoint listing files, by validating the that previously uploaded file is present in the response.

.. literalinclude:: ./app.py
   :lines: 165-182

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
   :header: Step 4: Implement tests for the File Server component with pytest
   :description: Create an App to validate the upload and list files endpoints
   :col_css: col-md-4
   :button_link: file_server_step_4.html
   :height: 180
   :tag: Intermediate

.. raw:: html

        </div>
    </div>
