
*************************************
Everything about LightningWork Status
*************************************

Statuses indicate transition points in the life of a LightningWork (Work) and contain metadata.

The different stages are:

.. code-block:: python

    class WorkStageStatus:
        NOT_STARTED = "not_started"
        STOPPED = "stopped"
        PENDING = "pending"
        RUNNING = "running"
        SUCCEEDED = "succeeded"
        FAILED = "failed"

And a single status is as follows:

.. code-block:: python

    @dataclass
    class WorkStatus:
        stage: WorkStageStatus
        timestamp: float
        reason: Optional[str] = None
        message: Optional[str] = None
        count: int = 1


On creation, the Work's status flags all evaluate to ``False`` (in particular ``has_started``) and when calling ``work.run`` in your Lightning Flow (Flow),
the Work transitions from ``is_pending`` to ``is_running`` and then to ``has_succeeded`` if everything went well or ``has_failed`` otherwise.

.. code-block:: python

    from time import sleep
    import lightning as L


    class Work(L.LightningWork):
        def run(self, value: int):
            sleep(1)
            if value == 0:
                return
            raise Exception(f"The provided value was {value}")


    class Flow(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.work = Work(raise_exception=False)
            self.counter = 0

        def run(self):
            if not self.work.has_started:
                print("NOT STARTED")

            elif self.work.is_pending:
                print("PENDING")

            elif self.work.is_running:
                print("RUNNING")

            elif self.work.has_succeeded:
                print("SUCCESS")

            elif self.work.has_failed:
                print("FAILED")

            elif self.work.has_stopped:
                print("STOPPED")
                self.stop()

            print(self.work.status)
            self.work.run(self.counter)
            self.counter += 1


    app = L.LightningApp(Flow())

Run this app as follows:

.. code-block:: bash

    lightning run app test.py > app_log.txt

And here is the expected output inside ``app_log.txt`` and as expected,
we are observing the following transition ``has_started``, ``is_pending``, ``is_running``, ``has_succeeded``, ``is_running`` and ``has_failed``

.. code-block:: console

    NOT STARTED
    WorkStatus(stage='not_started', timestamp=1653498225.18468, reason=None, message=None, count=1)
    PENDING
    WorkStatus(stage='pending', timestamp=1653498225.217413, reason=None, message=None, count=1)
    PENDING
    WorkStatus(stage='pending', timestamp=1653498225.217413, reason=None, message=None, count=1)
    PENDING
    ...
    PENDING
    WorkStatus(stage='pending', timestamp=1653498225.217413, reason=None, message=None, count=1)
    PENDING
    WorkStatus(stage='pending', timestamp=1653498225.217413, reason=None, message=None, count=1)
    RUNNING
    WorkStatus(stage='running', timestamp=1653498228.825194, reason=None, message=None, count=1)
    ...
    SUCCESS
    WorkStatus(stage='succeeded', timestamp=1653498229.831793, reason=None, message=None, count=1)
    SUCCESS
    WorkStatus(stage='succeeded', timestamp=1653498229.831793, reason=None, message=None, count=1)
    SUCCESS
    WorkStatus(stage='succeeded', timestamp=1653498229.831793, reason=None, message=None, count=1)
    RUNNING
    WorkStatus(stage='running', timestamp=1653498229.846451, reason=None, message=None, count=1)
    RUNNING
    ...
    WorkStatus(stage='running', timestamp=1653498229.846451, reason=None, message=None, count=1)
    RUNNING
    WorkStatus(stage='running', timestamp=1653498229.846451, reason=None, message=None, count=1)
    FAILED
    WorkStatus(stage='failed', timestamp=1653498230.852565, reason='user_exception', message='The provided value was 1', count=1)
    FAILED
    WorkStatus(stage='failed', timestamp=1653498230.852565, reason='user_exception', message='The provided value was 1', count=1)
    FAILED
    WorkStatus(stage='failed', timestamp=1653498230.852565, reason='user_exception', message='The provided value was 1', count=1)
    FAILED
    WorkStatus(stage='failed', timestamp=1653498230.852565, reason='user_exception', message='The provided value was 1', count=1)
    ...

In order to access all statuses:

.. code-block:: python

    from time import sleep
    import lightning as L


    class Work(L.LightningWork):
        def run(self, value: int):
            sleep(1)
            if value == 0:
                return
            raise Exception(f"The provided value was {value}")


    class Flow(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.work = Work(raise_exception=False)
            self.counter = 0

        def run(self):
            print(self.statuses)
            self.work.run(self.counter)
            self.counter += 1


    app = L.LightningApp(Flow())


Run this app as follows:

.. code-block:: bash

    lightning run app test.py > app_log.txt

And here is the expected output inside ``app_log.txt``:


.. code-block:: console

    # First execution with value = 0

    []
    [WorkStatus(stage='pending', timestamp=1653498622.252016, reason=None, message=None, count=1)]
    ...
    [WorkStatus(stage='pending', timestamp=1653498622.252016, reason=None, message=None, count=1)]
    [WorkStatus(stage='pending', timestamp=1653498622.252016, reason=None, message=None, count=1)]
    [WorkStatus(stage='pending', timestamp=1653498622.252016, reason=None, message=None, count=1), WorkStatus(stage='running', timestamp=1653498626.185683, reason=None, message=None, count=1)]
    [WorkStatus(stage='pending', timestamp=1653498622.252016, reason=None, message=None, count=1), WorkStatus(stage='running', timestamp=1653498626.185683, reason=None, message=None, count=1)]
    ...
    [WorkStatus(stage='pending', timestamp=1653498622.252016, reason=None, message=None, count=1), WorkStatus(stage='running', timestamp=1653498626.185683, reason=None, message=None, count=1)]
    [WorkStatus(stage='pending', timestamp=1653498622.252016, reason=None, message=None, count=1), WorkStatus(stage='running', timestamp=1653498626.185683, reason=None, message=None, count=1)]
    [WorkStatus(stage='pending', timestamp=1653498622.252016, reason=None, message=None, count=1), WorkStatus(stage='running', timestamp=1653498626.185683, reason=None, message=None, count=1), WorkStatus(stage='succeeded', timestamp=1653498627.191053, reason=None, message=None, count=1)]
    [WorkStatus(stage='pending', timestamp=1653498622.252016, reason=None, message=None, count=1), WorkStatus(stage='running', timestamp=1653498626.185683, reason=None, message=None, count=1), WorkStatus(stage='succeeded', timestamp=1653498627.191053, reason=None, message=None, count=1)]
    [WorkStatus(stage='pending', timestamp=1653498622.252016, reason=None, message=None, count=1), WorkStatus(stage='running', timestamp=1653498626.185683, reason=None, message=None, count=1), WorkStatus(stage='succeeded', timestamp=1653498627.191053, reason=None, message=None, count=1)]

    # Second execution with value = 1

    [WorkStatus(stage='pending', timestamp=1653498627.204636, reason=None, message=None, count=1), WorkStatus(stage='running', timestamp=1653498627.205509, reason=None, message=None, count=1)]
    [WorkStatus(stage='pending', timestamp=1653498627.204636, reason=None, message=None, count=1), WorkStatus(stage='running', timestamp=1653498627.205509, reason=None, message=None, count=1)]
    ...
    [WorkStatus(stage='pending', timestamp=1653498627.204636, reason=None, message=None, count=1), WorkStatus(stage='running', timestamp=1653498627.205509, reason=None, message=None, count=1)]
    [WorkStatus(stage='pending', timestamp=1653498627.204636, reason=None, message=None, count=1), WorkStatus(stage='running', timestamp=1653498627.205509, reason=None, message=None, count=1), WorkStatus(stage='running', timestamp=1653498627.205509, reason=None, message=None, count=1), WorkStatus(stage='failed', timestamp=1653498628.210164, reason='user_exception', message='The provided value was 1', count=1)]
    [WorkStatus(stage='pending', timestamp=1653498627.204636, reason=None, message=None, count=1), WorkStatus(stage='running', timestamp=1653498627.205509, reason=None, message=None, count=1), WorkStatus(stage='running', timestamp=1653498627.205509, reason=None, message=None, count=1), WorkStatus(stage='failed', timestamp=1653498628.210164, reason='user_exception', message='The provided value was 1', count=1)]
