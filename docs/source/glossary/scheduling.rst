:orphan:

##########
Scheduling
##########

The Lightning Scheduling system makes it easy to schedule your components execution with any arbitrary conditions.


----

************************
Schedule your components
************************

The LightningFlow has a ``schedule`` method which can be used to schedule your components.

.. code-block:: python

    from lightning.app import LightningWork, LightningFlow
    from lightning.app.storage import Path


    class MyFlow(LightningFlow):

        def run(self):
            if self.schedule("hourly"):
                # run some code once every hour.

            if self.schedule("daily"):
                # run some code once day.

            if self.schedule("daily") and anything_else:
                # run some code once day if the anything else is also True.

            if self.schedule("2 4 * * mon,fri"):
                # defined with cron syntax, run some code at 04:02 on every Monday and Friday.

Learn more about the cron syntax `here <https://github.com/kiorky/croniter>`_

----

**************
Best Practices
**************

In the example above, the line ``self.schedule("hourly")`` will return ``True`` for a **single** flow execution every hour. Mathematically, this is known as a dirac.

1. Instantiate your component under the schedule method and run outside as follows:

.. code-block:: python

    from lightning.app import LightningFlow
    from lightning.app.structures import List

    class ScheduledDAG(LightningFlow):
        def __init__(self):
            super().__init__()
            self.list = List()

        def run(self):
            if self.schedule("hourly"):
                # dynamically instantiate
                # don't forget to always attach
                # your components to the flow !!!
                self.list.append(MyDAGFlow(...))

            # run all dags, but the completed ones
            # are cached and don't re-execute.
            for dag in self.list:
                dag.run()


2. Run a single work under the schedule with different arguments to have it re-run.

.. code-block:: python

    from lightning.app import LightningFlow
    from time import time

    class ScheduledDAG(LightningFlow):
        def __init__(self):
            super().__init__()
            self.data_processor = DataProcessorWork(...)

        def run(self):
            ...
            if self.schedule("hourly"):
                self.data_processor.run(trigger_time=time())


3. Capture the event in the state and execute your sequential works outside.

.. code-block:: python

    from lightning.app import LightningFlow
    from time import time

    class ScheduledDAG(LightningFlow):
        def __init__(self):
            super().__init__()
            self.should_execute = False
            self.data_processor = DataProcessorWork(...)
            self.training_work = KerasTrainingWork(...)

        def run(self):
            ...
            if self.schedule("hourly"):
                self.should_execute = True

            # Runs in 10 min
            if self.should_execute:
                # Runs in 5 min
                self.data_processor.run(trigger_time=time())
                if self.data_processor.has_succeeded:
                    # Runs in 5 min
                    self.training_work.run(self.data_processor.data)
                if self.training_work.has_succeeded:
                    self.should_execute = False

----

***********
Limitations
***********

As stated above, the schedule acts as a dirac and is **True** for a single flow execution.
Therefore, sequential works execution under the schedule won't work as they don't complete within a single flow execution.

Here is an example of something which **WON'T** work:

.. code-block:: python

    from lightning.app import LightningFlow
    from time import time

    class ScheduledDAG(LightningFlow):
        def __init__(self):
            super().__init__()
            self.data_processor = DataProcessorWork(...)
            self.training_work = KerasTrainingWork(...)

        def run(self):
            ...
            if self.schedule("hourly"):
                # This finishes 5 min later
                self.data_processor.run(trigger_time=time())
                if self.data_processor.has_succeeded:
                    # This will never be reached as the
                    # data processor will keep processing forever...
                    self.training_work.run(self.data_processor.data)

----

**************************
Frequently Asked Questions
**************************

- **Q: Can I use multiple nested scheduler?** No, as they might cancel themselves out, but you can capture the event of one to trigger the next one.

- **Q: Can I use any arbitrary logic to schedule?** Yes, this design enables absolute flexibility, but you need to be careful to avoid bad practices.

----

********
Examples
********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Build a DAG
   :description: Learn how to schedule a DAG execution
   :col_css: col-md-4
   :button_link: ../examples/dag/dag.html
   :height: 180
   :tag: Intermediate

.. raw:: html

        </div>
    </div>
