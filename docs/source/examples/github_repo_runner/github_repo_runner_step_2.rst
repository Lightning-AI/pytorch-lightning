:orphan:

******************************************************************
Step 2: Implement the PyTorch Lightning GithubRepoRunner Component
******************************************************************

The PyTorch Lightning GithubRepoRunner Component subclasses the GithubRepoRunner but tailors the execution experience to PyTorch Lightning.

As a matter of fact, this component adds two primary tailored features for PyTorch Lightning users:

* It injects dynamically a custom callback ``TensorboardServerLauncher`` in the PyTorch Lightning Trainer to start a tensorboard server so it can be exposed in Lightning App UI.

* Once the script has run, the ``on_after_run`` hook of the :class:`~lightning.app.components.python.tracer.TracerPythonScript` is invoked with the script globals, meaning we can collect anything we need. In particular, we are reloading the best model, torch scripting it, and storing its path in the state along side the best metric score.

Let's dive in on how to develop the component with the following code:

.. literalinclude:: ./app.py
    :lines: 75-136

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
