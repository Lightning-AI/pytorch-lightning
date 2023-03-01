:orphan:

#################################################################
Step 4: Implement tests for the File Server component with pytest
#################################################################

Let's create a simple App with our **File Server** and **File Server Test** components.

Once the File Server is up and running, we'll execute the **test_file_server** LightningWork and when both calls are successful, we exit the App using ``self._exit``.

.. literalinclude:: ./app.py
   :lines: 187-218


Simply create a ``test.py`` file with the following code and run ``pytest tests.py``:

.. literalinclude:: ./app.py
   :lines: 221-226

To test the App in the cloud, create a ``cloud_test.py`` file with the following code and run ``pytest cloud_test.py``.
Under the hood, we are using the end-to-end testing `playwright <https://playwright.dev/python/>`_ library, so you can interact with the UI.

.. literalinclude:: ./app.py
   :lines: 229-

----

********************
Test the application
********************

Clone the Lightning repo and run the following command:

.. code-block:: bash

   pytest docs/source/examples/file_server/app.py --capture=no -v

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

.. raw:: html

        </div>
    </div>

----

******************
Find more examples
******************

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Develop a DAG
   :description: Create a dag pipeline
   :col_css: col-md-4
   :button_link: ../dag/dag.html
   :height: 150
   :tag: Intermediate

.. displayitem::
   :header: Develop a Github Repo Script Runner
   :description: Run any script on github in the cloud
   :col_css: col-md-4
   :button_link: ../github_repo_runner/github_repo_runner.html
   :height: 150
   :tag: Intermediate


.. displayitem::
   :header: Develop a HPO Sweeper
   :description: Train multiple models with different parameters
   :col_css: col-md-4
   :button_link: ../hpo/hpo.html
   :height: 150
   :tag: Intermediate

.. displayitem::
   :header: Develop a Model Server
   :description: Serve multiple models with different parameters
   :col_css: col-md-4
   :button_link: ../model_server_app/model_server_app.html
   :height: 150
   :tag: Intermediate

.. raw:: html

        </div>
    </div>
