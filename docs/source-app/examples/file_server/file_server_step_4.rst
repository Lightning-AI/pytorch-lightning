:orphan:

************************************************************
4. Implement tests for the File Server component with pytest
************************************************************

Let's create a simple Lightning App (App) with our **File Server** and the **File Server Test** components.

Once the File Server is up and running, we'll execute the **test_file_server** LightningWork and when both calls are successful, we exit the App using ``self._exit``.

.. literalinclude:: ./app.py
    :lines: 186-216


Simply create a ``test.py`` file with the following code and run ``pytest tests.py``

.. literalinclude:: ./app.py
    :lines: 218-222

To test the App in the cloud, create a ``cloud_test.py`` file with the following code and run ``pytest cloud_test.py``. Under the hood, we are using the end-to-end testing `playwright <https://playwright.dev/python/>`_ library so you can interact with the UI.

.. literalinclude:: ./app.py
    :lines: 224-

----

********************
Test the application
********************

Clone the lightning repo and run the following command:

.. code-block:: bash

   pytest docs/source-app/examples/file_server/app.py --capture=no -v

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
   :header: Build a Github Repo Script Runner
   :description: Run any script on github in the cloud
   :col_css: col-md-4
   :button_link: ../github_repo_runner/github_repo_runner.html
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
