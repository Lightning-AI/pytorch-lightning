.. PyTorch-Lightning documentation master file, created by
   sphinx-quickstart on Fri Nov 15 07:48:22 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ⚡ PyTorch Lightning
===============================

.. twocolumns::
   :left:
      .. image:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/mov.gif
         :alt: Animation showing how to convert a standard training loop to a Lightning loop
   :right:
      PyTorch Lightning is the deep learning framework for professional AI researchers and machine learning engineers who need maximal flexibility without sacrificing performance at scale.
      Lightning evolves with you as your projects go from idea to paper/production.

.. raw:: html

   <div class="row" style='font-size: 14px'>
      <div class='col-md-6'>
      </div>
      <div class='col-md-6'>

.. join_slack::
   :align: center
   :margin: 0

.. raw:: html

      </div>
   </div>


.. raw:: html

   <hr class="docutils" style="margin: 50px 0 50px 0">


Install Lightning
-----------------


.. raw:: html

   <div class="row" style='font-size: 16px'>
      <div class='col-md-6'>

Pip users

.. code-block:: bash

    pip install pytorch-lightning

.. raw:: html

      </div>
      <div class='col-md-6'>

Conda users

.. code-block:: bash

    conda install pytorch-lightning -c conda-forge

.. raw:: html

      </div>
   </div>

Or read the `advanced install guide <starter/installation.html>`_

.. raw:: html

   <hr class="docutils" style="margin: 50px 0 50px 0">

Get Started
-----------

.. raw:: html

    <div class="tutorials-callout-container">
        <div class="row">

.. Add callout items below this line

.. customcalloutitem::
   :description: Learn the 7 key steps of a typical Lightning workflow.
   :header: Lightning in 15 minutes
   :button_link:  starter/introduction.html

.. customcalloutitem::
   :description: Learn how to benchmark PyTorch Lightning.
   :header: Benchmarking
   :button_link: benchmarking/benchmarks.html

.. raw:: html

        </div>
    </div>

.. End of callout item section

.. raw:: html

   <hr class="docutils" style="margin: 50px 0 50px 0">

Current Lightning Users
-----------------------

.. raw:: html

    <div class="tutorials-callout-container">
        <div class="row">

.. Add callout items below this line

.. customcalloutitem::
   :description: Learn Lightning in small bites at 4 levels of expertise: Introductory, intermediate, advanced and expert.
   :header: Level Up!
   :button_link:  expertise_levels.html

.. customcalloutitem::
   :description: Detailed description of API each package. Assumes you already have basic Lightning knowledge.
   :header: API Reference
   :button_link: api_references.html

.. customcalloutitem::
   :description: From NLP, Computer vision to RL and meta learning - see how to use Lightning in ALL research areas.
   :header: Hands-on Examples
   :button_link: tutorials.html

.. customcalloutitem::
   :description: Learn how to do everything from hyperparameters sweeps to cloud training to Pruning and Quantization with Lightning.
   :header: Common Workflows
   :button_link: common_usecases.html

.. customcalloutitem::
   :description: Convert your current code to Lightning
   :header: Convert code to PyTorch Lightning
   :button_link: starter/converting.html


.. raw:: html

        </div>
    </div>

.. End of callout item section

.. raw:: html

   <div style="display:none">

.. toctree::
   :maxdepth: 1
   :name: start
   :caption: Get Started

   starter/introduction
   starter/installation


.. toctree::
   :maxdepth: 2
   :name: levels
   :caption: Level Up

   levels/core_skills
   levels/intermediate
   levels/advanced
   levels/expert

.. toctree::
   :maxdepth: 2
   :name: pl_docs
   :caption: Core API

   common/lightning_module
   common/trainer

.. toctree::
   :maxdepth: 2
   :name: api
   :caption: API Reference

   api_references

.. toctree::
   :maxdepth: 1
   :name: Common Workflows
   :caption: Common Workflows

   Avoid overfitting <common/evaluation>
   model/build_model.rst
   common/hyperparameters
   common/progress_bar
   deploy/production
   advanced/training_tricks
   cli/lightning_cli
   tuning/profiler
   Manage experiments <visualize/logging_intermediate>
   Organize existing PyTorch into Lightning <starter/converting>
   clouds/cluster
   Save and load model progress <common/checkpointing>
   Save memory with half-precision <common/precision>
   Training over the internet <strategies/collaborative_training>
   advanced/model_parallel
   clouds/cloud_training
   Train on single or multiple GPUs <accelerators/gpu>
   Train on single or multiple HPUs <accelerators/hpu>
   Train on single or multiple IPUs <accelerators/ipu>
   Train on single or multiple TPUs <accelerators/tpu>
   Use a pretrained model <advanced/pretrained>
   model/own_your_loop

.. toctree::
   :maxdepth: 1
   :name: Glossary
   :caption: Glossary

   Accelerators <extensions/accelerator>
   Callback <extensions/callbacks>
   Checkpointing <common/checkpointing>
   Cluster <clouds/cluster>
   Cloud checkpoint <common/checkpointing_advanced>
   Console Logging <common/console_logs>
   Debugging <debug/debugging>
   Early stopping <common/early_stopping>
   Experiment manager (Logger) <visualize/experiment_managers>
   Fault tolerant training  <clouds/fault_tolerant_training>
   Finetuning <advanced/finetuning>
   Flash <https://lightning-flash.readthedocs.io/en/stable/>
   Grid AI <clouds/cloud_training>
   GPU <accelerators/gpu>
   Half precision <common/precision>
   HPU <accelerators/hpu>
   Inference <deploy/production_intermediate>
   IPU <accelerators/ipu>
   Lightning CLI <cli/lightning_cli>
   Lightning Lite <model/build_model_expert>
   LightningDataModule <data/datamodule>
   LightningModule <common/lightning_module>
   Lightning Transformers <https://pytorch-lightning.readthedocs.io/en/stable/ecosystem/transformers.html>
   Log <visualize/loggers>
   Loops <extensions/loops>
   TPU <accelerators/tpu>
   Metrics <https://torchmetrics.readthedocs.io/en/stable/>
   Model <model/build_model.rst>
   Model Parallel <advanced/model_parallel>
   Collaborative Training <strategies/collaborative_training>
   Plugins <extensions/plugins>
   Progress bar <common/progress_bar>
   Production <deploy/production_advanced>
   Predict <deploy/production_basic>
   Pretrained models <advanced/pretrained>
   Profiler <tuning/profiler>
   Pruning and Quantization <advanced/pruning_quantization>
   Remote filesystem and FSSPEC <common/remote_fs>
   Strategy <extensions/strategy>
   Strategy registry <advanced/strategy_registry>
   Style guide <starter/style_guide>
   Sweep <clouds/run_intermediate>
   SWA <advanced/training_tricks>
   SLURM <clouds/cluster_advanced>
   Transfer learning <advanced/transfer_learning>
   Trainer <common/trainer>
   Torch distributed <clouds/cluster_intermediate_2>

.. toctree::
   :maxdepth: 1
   :name: Hands-on Examples
   :caption: Hands-on Examples
   :glob:

   notebooks/**/*
   PyTorch Lightning 101 class <https://www.youtube.com/playlist?list=PLaMu-SDt_RB5NUm67hU2pdE75j6KaIOv2>
   From PyTorch to PyTorch Lightning [Blog] <https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09>
   From PyTorch to PyTorch Lightning [Video] <https://www.youtube.com/watch?v=QHww1JH7IDU>

.. toctree::
   :maxdepth: 1
   :name: Community
   :caption: Community

   generated/CODE_OF_CONDUCT.md
   generated/CONTRIBUTING.md
   generated/BECOMING_A_CORE_CONTRIBUTOR.md
   governance
   generated/CHANGELOG.md

.. raw:: html

   </div>
