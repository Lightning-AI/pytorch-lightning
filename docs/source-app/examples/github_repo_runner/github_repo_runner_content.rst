
*********
Objective
*********

Create a simple application where users can enter information in a UI to run a given PyTorch Lightning Script from a given Github Repo with optionally some extra python requirements and arguments.

Furthermore, the users should be able to monitor their training progress in real-time, view the logs, and get the best-monitored metric and associated checkpoint for their models.

----

*****************
Final Application
*****************

Here is a recording of the final application built in this example. The example is around 200 lines in total and should give you a great foundation to build your own Lightning App.

.. raw:: html

   <video id="background-video" autoplay loop muted controls poster="https://pl-flash-data.s3.amazonaws.com/assets_lightning/github_app.png" width="100%">
      <source src="https://pl-flash-data.s3.amazonaws.com/assets_lightning/github_app.mp4" type="video/mp4" width="100%">
   </video>

----

*************
System Design
*************

In order to create such application, we need to build several components:

* A GithubRepoRunner Component that clones a repo, runs a specific script with provided arguments and collect logs.

* A PyTorch Lightning GithubRepoRunner Component that augments the GithubRepoRunner component to track PyTorch Lightning Trainer.

* A UI for the users to provide to trigger dynamically a new execution.

* A Flow to dynamically create GithubRepoRunner once a user submits information from the UI.

Let's dive in on how to create such a component.

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
   :header: 4. Implement the UI with StreamLit
   :description: Several pages application
   :col_css: col-md-4
   :button_link: github_repo_runner_step_4.html
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
