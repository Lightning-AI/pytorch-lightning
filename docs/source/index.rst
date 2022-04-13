.. PyTorch-Lightning documentation master file, created by
   sphinx-quickstart on Fri Nov 15 07:48:22 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ⚡ PyTorch Lightning 
==============================

.. raw:: html

   <div class="row" style='font-size: 14px'>
      <div class='col-md-6'>

.. image:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/mov.gif
    :alt: Animation showing how to convert a standard training loop to a Lightning loop


.. raw:: html

      </div>
      <div class='col-md-6'>

PyTorch Lightning is the deep learning framework for professional AI researchers and machine learning engineers who need maximal flexibility without sacrificing performance at scale.

.. raw:: html

      </div>
   </div>


.. raw:: html

   <div class='row'>
      <a class="sc-frDJqD lkTDih menu-link" href="https://join.slack.com/t/gridai-community/shared_invite/zt-ozqiwuif-UYK6rZGVmTTpMfPcVSdicg" target="_blank">
         <div class="sc-bdVaJa iMxqiR icon" data-icon="slackLogo" style="width: 20px; height: 20px; color: var(--gray-50);">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
               <path d="M5.042 15.165a2.528 2.528 0 0 1-2.52 2.523A2.528 2.528 0 0 1 0 15.165a2.527 2.527 0 0 1 2.522-2.52h2.52v2.52zM6.313 15.165a2.527 2.527 0 0 1 2.521-2.52 2.527 2.527 0 0 1 2.521 2.52v6.313A2.528 2.528 0 0 1 8.834 24a2.528 2.528 0 0 1-2.521-2.522v-6.313zM8.834 5.042a2.528 2.528 0 0 1-2.521-2.52A2.528 2.528 0 0 1 8.834 0a2.528 2.528 0 0 1 2.521 2.522v2.52H8.834zM8.834 6.313a2.528 2.528 0 0 1 2.521 2.521 2.528 2.528 0 0 1-2.521 2.521H2.522A2.528 2.528 0 0 1 0 8.834a2.528 2.528 0 0 1 2.522-2.521h6.312zM18.956 8.834a2.528 2.528 0 0 1 2.522-2.521A2.528 2.528 0 0 1 24 8.834a2.528 2.528 0 0 1-2.522 2.521h-2.522V8.834zM17.688 8.834a2.528 2.528 0 0 1-2.523 2.521 2.527 2.527 0 0 1-2.52-2.521V2.522A2.527 2.527 0 0 1 15.165 0a2.528 2.528 0 0 1 2.523 2.522v6.312zM15.165 18.956a2.528 2.528 0 0 1 2.523 2.522A2.528 2.528 0 0 1 15.165 24a2.527 2.527 0 0 1-2.52-2.522v-2.522h2.52zM15.165 17.688a2.527 2.527 0 0 1-2.52-2.523 2.526 2.526 0 0 1 2.52-2.52h6.313A2.527 2.527 0 0 1 24 15.165a2.528 2.528 0 0 1-2.522 2.523h-6.313z" fill="currentColor">
               </path>
            </svg>
         </div>
         <button type="button" data-intercom-target="Get Help" class="sc-bxivhb lhLdPd new-btn xl primary textLink iconSize rightIconSize">
            <span>Get Help</span>
         </button>
      </a>
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

Getting Started
---------------

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
   :header: API References
   :button_link: api_references.html

.. customcalloutitem::
   :description: From NLP, Computer vision to RL and meta learning - see how to use Lightning in ALL research areas.
   :header: Hands-on Examples
   :button_link: tutorials.html

.. customcalloutitem::
   :description: Learn how to do everything from hyperparameters sweeps to cloud training to Pruning and Quantization with Lightning.
   :header: Concepts Glossary
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
   :caption: Getting Started

   starter/introduction


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
   :caption: API References

   api_references

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
   :name: Concepts Glossary
   :caption: Concepts Glossary

   accelerators/gpu
   accelerators/hpu
   accelerators/ipu
   accelerators/tpu
   clouds/cloud_training
   common/checkpointing
   clouds/cluster
   common/console_logs
   common/debugging
   build_model/build_model.rst
   deploy/production
   common/early_stopping
   advanced/training_tricks
   common/evaluation
   visualize_experiments/experiment_managers
   advanced/fault_tolerant_training
   tuning/profiler
   common/hyperparameters
   common/lightning_cli
   advanced/model_parallel
   precision/precision
   common/optimization
   common/progress_bar
   advanced/pruning_quantization
   common/remote_fs
   advanced/strategy_registry
   visualize_experiments/loggers
   advanced/transfer_learning

.. toctree::
   :maxdepth: 1
   :name: Community
   :caption: Community

   generated/CODE_OF_CONDUCT.md
   generated/CONTRIBUTING.md
   generated/BECOMING_A_CORE_CONTRIBUTOR.md
   governance
   generated/CHANGELOG.md


   * :ref:`genindex`
   * :ref:`search`

.. raw:: html

   </div>