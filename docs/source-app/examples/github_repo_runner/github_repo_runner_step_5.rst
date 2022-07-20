:orphan:

******************************
5. Putting everything together
******************************

Let's dive in on how to create such a component with the code below.

.. literalinclude:: ./app.py
    :lines: 277-


*******************
Run the application
*******************

Clone the lightning repo and run the following command:

.. code-block:: bash

   lightning run app docs/source-app/examples/github_repo_runner/app.py

Add **--cloud** to run this application in the cloud.

.. code-block:: bash

   lightning run app docs/source-app/examples/github_repo_runner/app.py --cloud

----

******************
Find more examples
******************

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Build a DAG
   :description: Create a dag pipeline
   :col_css: col-md-4
   :button_link: ../dag/dag.html
   :height: 150
   :tag: Intermediate

.. displayitem::
   :header: Build a File Server
   :description: Train multiple models with different parameters
   :col_css: col-md-4
   :button_link: ../file_server/file_server.html
   :height: 150
   :tag: Intermediate

.. displayitem::
   :header: Build a HPO Sweeper
   :description: Train multiple models with different parameters
   :col_css: col-md-4
   :button_link: ../hpo/hpo.html
   :height: 150
   :tag: Intermediate

.. displayitem::
   :header: Build a Model Server
   :description: Serve multiple models with different parameters
   :col_css: col-md-4
   :button_link: ../model_server/model_server.html
   :height: 150
   :tag: Intermediate

.. raw:: html

        </div>
    </div>
