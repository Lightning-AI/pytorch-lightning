:orphan:

################################################################
Step 2: Implement the File Server upload and list_files methods
################################################################

Let's dive in on how to implement these methods.

***************************
Implement the upload method
***************************

In this method, we are creating a stream between the uploaded file and the uploaded file stored on the file server disk.

Once the file is uploaded, we are putting the file into the :class:`~lightning.app.storage.drive.Drive`, so it becomes persistent and accessible to all Components.

.. literalinclude:: ./app.py
    :lines: 12, 51-99
    :emphasize-lines: 49

*******************************
Implement the fist_files method
*******************************

First, in this method, we get the file in the file server filesystem, if available in the Drive. Once done, we list the the files under the provided paths and return the results.

.. literalinclude:: ./app.py
    :lines: 12, 100-130
    :emphasize-lines: 9


*******************
Implement utilities
*******************

.. literalinclude:: ./app.py
    :lines: 12, 43-49

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
