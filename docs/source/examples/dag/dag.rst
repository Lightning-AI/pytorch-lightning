:orphan:

######################################
Develop a Directed Acyclic Graph (DAG)
######################################

.. _dag_example:

**Audience:** Users coming from MLOps to Lightning Apps, looking for more flexibility.

A typical ML training workflow can be implemented with a simple DAG.

Below is a pseudo-code using the lightning framework that uses a LightningFlow to orchestrate the serial workflow: process data, train a model, and serve the model.

.. code-block:: python

    import lightning as L

    class DAGFlow(L.LightningFlow):

        def __init__(self):
            super().__init__()
            self.processor = DataProcessorWork(...)
            self.train_work = TrainingWork(...)
            self.serve_work = ServeWork(...)

        def run(self):
            self.processor.run(...)
            self.train_work.run(...)
            self.serve_work.run(...)

Below is a pseudo-code to run several works in parallel using a built-in :class:`~lightning.app.structures.Dict`.

.. code-block:: python

    import lightning as L

    class DAGFlow(L.LightningFlow):

        def __init__(self):
            super().__init__()
            ...
            self.train_works = L.structures.Dict(**{
                "1": TrainingWork(..., parallel=True),
                "2": TrainingWork(..., parallel=True),
                "3": TrainingWork(..., parallel=True),
                ...
                })
            ...

        def run(self):
            self.processor.run(...)

            # The flow runs through them all, so we need to guard self.serve_work.run
            for work in self.train_works.values():
                work.run(...)

            # Wait for all to have finished without errors.
            if not all(w.has_succeeded for w in self.train_works):
                continue

            self.serve_work.run(...)

----

**********
Next Steps
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Scheduled DAG with pandas and sklearn from scratch.
   :description: DAG example in pure Lightning.
   :col_css: col-md-4
   :button_link: dag_from_scratch.html
   :height: 180
   :tag: intermediate
