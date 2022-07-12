:orphan:

**********************************
4. Implement the UI with StreamLit
**********************************

In step 3, we have implemented a flow that dynamically creates a Work when a new request is added to the requests list.

From the UI, we create 3 pages with `StreamLit <https://streamlit.io/>`_:

* **Page 1**: Create a form to add a new request to the flow state **requests**.

* **Page 2**: Iterate through all the requests and display associated information.

* **Page 3**: Display the entire App State.

****************
Render All Pages
****************

.. literalinclude:: ./app.py
    :lines: 263-274


******
Page 1
******

.. literalinclude:: ./app.py
    :lines: 189-231
    :emphasize-lines: 43

******
Page 2
******

.. literalinclude:: ./app.py
    :lines: 233-255

******
Page 3
******

.. literalinclude:: ./app.py
    :lines: 257-261

----

********
Tutorial
********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: 1. Implement the GithubRepoRunner Component
   :description: Clone and execute script from a GitHub Repo.
   :col_css: col-md-4
   :button_link: github_repo_runner_step_1.html
   :height: 180
   :tag: Intermediate

.. displayitem::
   :header: 2. Implement the PyTorch Lightning GithubRepoRunner Component
   :description: Automate PyTorch Lightning execution
   :col_css: col-md-4
   :button_link: github_repo_runner_step_2.html
   :height: 180
   :tag: Advanced

.. displayitem::
   :header: 3. Implement the Flow to manage user requests
   :description: Dynamically create GithubRepoRunner
   :col_css: col-md-4
   :button_link: github_repo_runner_step_3.html
   :height: 180
   :tag: Intermediate

.. displayitem::
   :header: 5. Putting everything together
   :description:
   :col_css: col-md-4
   :button_link: github_repo_runner_step_5.html
   :height: 180
   :tag: Intermediate

.. raw:: html

        </div>
    </div>
