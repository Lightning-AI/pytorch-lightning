:orphan:

########################
Handling App Exceptions
########################

**Audience:** Users who want to know how to implement app where errors are handled.

**Level:** Advanced

----

*************************************************
Why should I care about handling app exceptions ?
*************************************************

Imagine you are creating an application where your team can launch model training by providing their own Github Repo any time they want.

As the application admin, you don't want the application to go down if their code has a bug and breaks.

Instead, you would like the work to capture the exception and surface this to the users on failures.

****************************************
How can I configure exception handling ?
****************************************


The LightningWork accepts an argument **raise_exception** which is **True** by default. This aligns with Python default behaviors.

However, for the user case stated above, we want to capture the work exceptions. This is done by providing ``raise_exception=False`` to the work ``__init__`` method.

.. code-block:: python

    MyCustomWork(raise_exception=False)  # <== HERE: The exception is captured.

    # Default behavior
    MyCustomWork(raise_exception=True)  # <== HERE: The exception is raised within the flow and terminates the app


And you can customize this behavior by overriding the ``on_exception`` hook to the Lightning Work.

.. code-block:: python

    import lightning as L


    class MyCustomWork(L.LightningWork):
        def on_exception(self, exception: Exception):
            # do something when an exception is triggered.
            pass


*******************
Application Example
*******************

This is the pseudo-code for the application described above.

.. code-block:: python

    import lightning_app as lapp


    class RootFlow(lapp.LightningFlow):
        def __init__(self):
            super().__init__()
            self.user_jobs = lapp.structures.Dict()
            self.requested_jobs = []

        def run(self):
            for request in self.requested_jobs:
                job_id = request["id"]
                if job_id not in self.user_jobs:
                    # Note: The `GithubRepoLauncher` doesn't exist yet.
                    self.user_jobs[job_id] = GithubRepoLauncher(
                        **request,
                        raise_exception=False,  # <== HERE: The exception is captured.
                    )
                self.user_jobs[job_id].run()

                if self.user_jobs[job_id].status.stage == "failed" and "printed" not in request:
                    print(self.user_jobs[job_id].status)  # <== HERE: Print the user exception.
                    request["printed"] = True
