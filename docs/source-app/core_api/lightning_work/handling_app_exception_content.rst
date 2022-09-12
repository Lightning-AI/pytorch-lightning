
***************************************************
What handling Lightning App exceptions does for you
***************************************************

Imagine you are creating a Lightning App (App) where your team can launch model training by providing their own Github Repo any time they want.

As the App admin, you don't want the App to go down if their code has a bug and breaks.

Instead, you would like the LightningWork (Work) to capture the exception and present the issue to users.

----

****************************
Configure exception handling
****************************

The LightningWork (Work) accepts an argument **raise_exception** which is **True** by default. This aligns with Python default behaviors.

However, for the user case stated in the previous section, we want to capture the Work exceptions. This is done by providing ``raise_exception=False`` to the work ``__init__`` method.

.. code-block:: python

    import lightning as L

    MyCustomWork(raise_exception=False)  # <== HERE: The exception is captured.

    # Default behavior
    MyCustomWork(raise_exception=True)  # <== HERE: The exception is raised within the flow and terminates the app


And you can customize this behavior by overriding the ``on_exception`` hook to the Work.

.. code-block:: python

    import lightning as L

    class MyCustomWork(L.LightningWork):

        def on_exception(self, exception: Exception):
            # do something when an exception is triggered.

----

**************************
Exception handling example
**************************

This is the pseudo-code for the application described above.

.. code-block:: python

    import lightning as L

    class RootFlow(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.user_jobs = L.structures.Dict()
            self.requested_jobs = []

        def run(self):
            for request in self.requested_jobs:
                job_id = request["id"]
                if job_id not in self.user_jobs:
                    # Note: The `GithubRepoLauncher` doesn't exist yet.
                    self.user_jobs[job_id] = GithubRepoLauncher(
                        **request,
                        raise_exception=False, # <== HERE: The exception is captured.
                    )
                self.user_jobs[job_id].run()

                if self.user_jobs[job_id].status.stage == "failed" and "printed" not in request:
                    print(self.user_jobs[job_id].status) # <== HERE: Print the user exception.
                    request["printed"] = True
