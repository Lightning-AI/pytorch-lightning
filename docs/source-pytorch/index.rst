Welcome to ⚡ PyTorch Lightning
===============================

.. twocolumns::
   :left:
      .. image:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/mov.gif
         :alt: Animation showing how to convert standard training code to Lightning
   :right:
      PyTorch Lightning is the deep learning framework for professional AI researchers and machine learning engineers who need maximal flexibility without sacrificing performance at scale.
      Lightning evolves with you as your projects go from idea to paper/production.

.. raw:: html

   <div class="row" style='font-size: 14px'>
      <div class='col-md-6'>
      </div>
      <div class='col-md-6'>

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

    pip install lightning

.. raw:: html

      </div>
      <div class='col-md-6'>

Conda users

.. code-block:: bash

    conda install lightning -c conda-forge

.. raw:: html

      </div>
   </div>

Or read the `advanced install guide <starter/installation.html>`_

You can find the list of supported PyTorch versions in our :ref:`compatibility matrix <versioning:Compatibility matrix>`.

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
   :description: Learn how to do everything from hyper-parameters sweeps to cloud training to Pruning and Quantization with Lightning.
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
   :caption: Home

   starter/introduction
   Install <starter/installation>
   upgrade/migration_guide


.. toctree::
   :maxdepth: 2
   :name: levels
   :caption: Level Up

   levels/core_skills
   levels/intermediate
   levels/advanced
   levels/expert

.. toctree::
   :maxdepth: 1
   :name: pl_docs
   :caption: Core API

   common/lightning_module
   common/trainer

.. toctree::
   :maxdepth: 1
   :name: api
   :caption: Optional API

   api_references

.. toctree::
   :maxdepth: 1
   :name: More
   :caption: More

   Community <community/index>
   Examples <tutorials>
   Glossary <glossary/index>
   How to <common/index>


.. raw:: html

   </div>

.. PyTorch-Lightning documentation master file, created by
   sphinx-quickstart on Fri Nov 15 07:48:22 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
