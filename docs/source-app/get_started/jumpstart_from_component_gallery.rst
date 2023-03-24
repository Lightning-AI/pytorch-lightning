:orphan:

########################################
Add Component made by others to your App
########################################

.. _jumpstart_from_component_gallery:

Anyone can build components for their own use case and promote them on the `Component Gallery <https://lightning.ai/components>`_.

In return, you can benefit from the work of others and add new functionalities to your Apps with minimal effort.


*************
User Workflow
*************

#. Visit the `Component Gallery <https://lightning.ai/components>`_ and look for a Component close to something you want to do.

    .. raw:: html

       <br />

#. Check out the code for inspiration or simply install the component from PyPi and use it.

----

*************
Success Story
*************

The default `Train and Demo Application <https://github.com/Lightning-AI/lightning-quick-start>`_ trains a PyTorch Lightning
model and then starts a demo with `Gradio <https://gradio.app/>`_.

.. code-block:: python

    import os.path as ops
    import lightning as L
    from quick_start.components import PyTorchLightningScript, ImageServeGradio


    class TrainDeploy(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.train_work = PyTorchLightningScript(
                script_path=ops.join(ops.dirname(__file__), "./train_script.py"),
                script_args=["--trainer.max_epochs=5"],
            )

            self.serve_work = ImageServeGradio(L.CloudCompute("cpu"))

        def run(self):
            # 1. Run the python script that trains the model
            self.train_work.run()

            # 2. when a checkpoint is available, deploy
            if self.train_work.best_model_path:
                self.serve_work.run(self.train_work.best_model_path)

        def configure_layout(self):
            tab_1 = {"name": "Model training", "content": self.train_work}
            tab_2 = {"name": "Interactive demo", "content": self.serve_work}
            return [tab_1, tab_2]


    app = L.LightningApp(TrainDeploy())

However, someone who wants to use this Aop (maybe you) found `Lightning HPO <https://lightning.ai/component/BA2slXI093-Lightning%20HPO>`_
from browsing the `Component Gallery <https://lightning.ai/components>`_ and decided to give it a spin after checking the associated
`Github Repository <https://github.com/Lightning-AI/LAI-lightning-hpo-App>`_.

Once ``lightning_hpo`` installed, they improved the default App by easily adding HPO support to their project.

Here is the resulting App. It is almost the same code, but it's way more powerful now!

This is the power of `lightning.ai <https://lightning.ai/>`_ ecosystem 🔥⚡🔥

.. code-block:: python

    import os.path as ops
    import lightning as L
    from quick_start.components import PyTorchLightningScript, ImageServeGradio
    import optuna
    from optuna.distributions import LogUniformDistribution
    from lightning_hpo import Optimizer, BaseObjective


    class HPOPyTorchLightningScript(PyTorchLightningScript, BaseObjective):
        @staticmethod
        def distributions():
            return {"model.lr": LogUniformDistribution(0.0001, 0.1)}


    class TrainDeploy(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.train_work = Optimizer(
                script_path=ops.join(ops.dirname(__file__), "./train_script.py"),
                script_args=["--trainer.max_epochs=5"],
                objective_cls=HPOPyTorchLightningScript,
                n_trials=4,
            )

            self.serve_work = ImageServeGradio(L.CloudCompute("cpu"))

        def run(self):
            # 1. Run the python script that trains the model
            self.train_work.run()

            # 2. when a checkpoint is available, deploy
            if self.train_work.best_model_path:
                self.serve_work.run(self.train_work.best_model_path)

        def configure_layout(self):
            tab_1 = {"name": "Model training", "content": self.train_work.hi_plot}
            tab_2 = {"name": "Interactive demo", "content": self.serve_work}
            return [tab_1, tab_2]


    app = L.LightningApp(TrainDeploy())

----

**********
Next Steps
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Start from Ready-to-Run Template Apps
   :description: Jump-start your projects development
   :col_css: col-md-6
   :button_link: jumpstart_from_app_gallery.html
   :height: 180

.. displayitem::
   :header: Level-up your skills with Lightning Apps
   :description: From Basic to Advanced Skills
   :col_css: col-md-6
   :button_link: ../levels/basic/index.html
   :height: 180

.. raw:: html

      </div>
   </div>
   <br />
