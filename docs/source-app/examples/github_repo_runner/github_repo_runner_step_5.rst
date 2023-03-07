:orphan:

***************************
Step 5: Put it all together
***************************

Let's dive in on how to develop the component with the following code:

.. literalinclude:: ./app.py
    :lines: 287-

Run the application
^^^^^^^^^^^^^^^^^^^

Clone the Lightning repo and run the following command:

.. code-block:: bash

   lightning run app docs/source/examples/github_repo_runner/app.py

Add ``--cloud`` to run this application in the cloud.

.. code-block:: bash

   lightning run app docs/source/examples/github_repo_runner/app.py --cloud

----

**********************
More hands-on examples
**********************

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
   :header: Develop a File Server
   :description: Train multiple models with different parameters
   :col_css: col-md-4
   :button_link: ../file_server/file_server.html
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
   :button_link: ../model_server/model_server.html
   :height: 150
   :tag: Intermediate

.. raw:: html

        </div>
    </div>
