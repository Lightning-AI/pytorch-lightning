:orphan:

.. _profiler_expert:

######################################
Find bottlenecks in your code (expert)
######################################
**Audience**: Users who want to build their own profilers.

----

***********************
Build your own profiler
***********************
To build your own profiler, subclass :class:`~lightning.pytorch.profilers.profiler.Profiler`
and override some of its methods. Here is a simple example that profiles the first occurrence and total calls of each action:

.. code-block:: python

    from lightning.pytorch.profilers import Profiler
    from collections import defaultdict
    import time


    class ActionCountProfiler(Profiler):
        def __init__(self, dirpath=None, filename=None):
            super().__init__(dirpath=dirpath, filename=filename)
            self._action_count = defaultdict(int)
            self._action_first_occurrence = {}

        def start(self, action_name):
            if action_name not in self._action_first_occurrence:
                self._action_first_occurrence[action_name] = time.strftime("%m/%d/%Y, %H:%M:%S")

        def stop(self, action_name):
            self._action_count[action_name] += 1

        def summary(self):
            res = f"\nProfile Summary: \n"
            max_len = max(len(x) for x in self._action_count)

            for action_name in self._action_count:
                # generate summary for actions called more than once
                if self._action_count[action_name] > 1:
                    res += (
                        f"{action_name:<{max_len}s} \t "
                        + "self._action_first_occurrence[action_name]} \t "
                        + "{self._action_count[action_name]} \n"
                    )

            return res

        def teardown(self, stage):
            self._action_count = {}
            self._action_first_occurrence = {}
            super().teardown(stage=stage)

.. code-block:: python

    trainer = Trainer(profiler=ActionCountProfiler())
    trainer.fit(...)

----

**********************************
Profile custom actions of interest
**********************************
To profile a specific action of interest, reference a profiler in the LightningModule.

.. code-block:: python

    from lightning.pytorch.profilers import SimpleProfiler, PassThroughProfiler


    class MyModel(LightningModule):
        def __init__(self, profiler=None):
            self.profiler = profiler or PassThroughProfiler()

To profile in any part of your code, use the **self.profiler.profile()** function

.. code-block:: python

    class MyModel(LightningModule):
        def custom_processing_step(self, data):
            with self.profiler.profile("my_custom_action"):
                ...
            return data

Here's the full code:

.. code-block:: python

    from lightning.pytorch.profilers import SimpleProfiler, PassThroughProfiler


    class MyModel(LightningModule):
        def __init__(self, profiler=None):
            self.profiler = profiler or PassThroughProfiler()

        def custom_processing_step(self, data):
            with self.profiler.profile("my_custom_action"):
                ...
            return data


    profiler = SimpleProfiler()
    model = MyModel(profiler)
    trainer = Trainer(profiler=profiler, max_epochs=1)
