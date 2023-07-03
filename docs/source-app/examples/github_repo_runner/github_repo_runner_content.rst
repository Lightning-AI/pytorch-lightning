
********
Our Goal
********

Create a simple Lightning App (App) where users can enter information in a UI to run a given PyTorch Lightning Script from a given Github Repo with some optional extra Python requirements and arguments.

Users should be able to monitor their training progress in real-time, view the logs, and get the best monitored metric and associated checkpoint for their models.

----

Completed App
^^^^^^^^^^^^^

Here is a recording of the final application built in this example. The example is around 200 lines in total and should give you a great foundation to build your own Lightning App.

.. video:: https://pl-public-data.s3.amazonaws.com/assets_lightning/github_app.mp4
    :poster: "https://pl-public-data.s3.amazonaws.com/assets_lightning/github_app.png
    :width: 600
    :class: background-video
    :autoplay:
    :loop:
    :muted:

----

**********
App Design
**********

In order to develop the App, we need to build several components:

* A GithubRepoRunner Component that clones a repo, runs a specific script with provided arguments and collect logs.

* A PyTorch Lightning GithubRepoRunner Component that augments the GithubRepoRunner component to track PyTorch Lightning Trainer.

* A UI for the users to provide to trigger dynamically a new execution.

* A Flow to dynamically create GithubRepoRunner once a user submits information from the UI.

----

********
Tutorial
********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Step 1: Implement the GithubRepoRunner Component
   :description: Clone and execute script from a GitHub Repo.
   :col_css: col-md-4
   :button_link: github_repo_runner_step_1.html
   :height: 180
   :tag: Intermediate

.. displayitem::
   :header: Step 2: Implement the PyTorch Lightning GithubRepoRunner Component
   :description: Automate PyTorch Lightning execution
   :col_css: col-md-4
   :button_link: github_repo_runner_step_2.html
   :height: 180
   :tag: Advanced

.. displayitem::
   :header: Step 3: Implement the Flow to manage user requests
   :description: Dynamically create GithubRepoRunner
   :col_css: col-md-4
   :button_link: github_repo_runner_step_3.html
   :height: 180
   :tag: Intermediate


.. displayitem::
   :header: Step 4: Implement the UI with StreamLit
   :description: Several pages application
   :col_css: col-md-4
   :button_link: github_repo_runner_step_4.html
   :height: 180
   :tag: Intermediate


.. displayitem::
   :header: Step 5: Put it all together
   :description:
   :col_css: col-md-4
   :button_link: github_repo_runner_step_5.html
   :height: 180
   :tag: Intermediate

.. raw:: html

        </div>
    </div>
