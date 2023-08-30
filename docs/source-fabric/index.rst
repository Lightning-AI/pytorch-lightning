.. include:: links.rst

####################
Welcome to ⚡ Fabric
####################

.. twocolumns::
   :left:
      .. image:: https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/mov.gif
         :alt: Animation showing how to convert standard training code to Lightning
   :right:
      Fabric is the fast and lightweight way to scale PyTorch models without boilerplate code which allows you to
      easily switch from running on CPU to GPU (Apple Silicon, CUDA, ...), TPU, multi-GPU or even multi-node training.
      You can also leverage State-of-the-art distributed training strategies (DDP, FSDP, DeepSpeed)
      and mixed precision out of the box.

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

Or read the :doc:`advanced install guide <fundamentals/installation>`.

You can find our the list of supported PyTorch versions in our :ref:`compatibility matrix <versioning:Compatibility matrix>`.

.. raw:: html

   <hr class="docutils" style="margin: 50px 0 50px 0">


Why Fabric?
-----------

|
|

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/fabric/PyTorch-to-Fabric-Spectrum-2.svg
   :alt: Fabric spans across a large spectrum - from raw PyTorch all the way to high-level PyTorch Lightning
   :width: 100%

|
|

Fabric differentiates itself from a fully-fledged trainer like Lightning's `Trainer`_ in these key aspects:

**Fast to implement**
There is no need to restructure your code: Just change a few lines in the PyTorch script and you'll be able to leverage Fabric features.

**Maximum Flexibility**
Write your own training and/or inference logic down to the individual optimizer calls.
You aren't forced to conform to a standardized epoch-based training loop like the one in Lightning `Trainer`_.
You can do flexible iteration based training, meta-learning, cross-validation and other types of optimization algorithms without digging into framework internals.
This also makes it super easy to adopt Fabric in existing PyTorch projects to speed-up and scale your models without the compromise on large refactors.
Just remember: With great power comes a great responsibility.

**Maximum Control**
The Lightning `Trainer`_ has many built-in features to make research simpler with less boilerplate, but debugging it requires some familiarity with the framework internals.
In Fabric, everything is opt-in. Think of it as a toolbox: You take out the tools (Fabric functions) you need and leave the other ones behind.
This makes it easier to develop and debug your PyTorch code as you gradually add more features to it.
Fabric provides important tools to remove undesired boilerplate code (distributed, hardware, checkpoints, logging, ...), but leaves the design and orchestration fully up to you.


.. raw:: html

   <hr class="docutils" style="margin: 50px 0 50px 0">

Get Started
-----------

.. raw:: html

    <div class="tutorials-callout-container">
        <div class="row">

.. Add callout items below this line

.. customcalloutitem::
   :description: Detail introduction how to install this package.
   :header: Install Lightning Fabric
   :button_link:  fundamentals/installation.html

.. customcalloutitem::
   :description: Five easy steps to let ``Fabric`` scale your PyTorch models.
   :header: From PyTorch to Fabric
   :button_link:  fundamentals/convert.html

.. customcalloutitem::
   :description: Learn how to run your code on almost any accelerator HW.
   :header: Accelerate your code with Fabric
   :button_link:  fundamentals/accelerators.html

.. customcalloutitem::
   :description: Enable training large models on limited resources.
   :header: Save memory with lower precision
   :button_link:  fundamentals/precision.html

.. customcalloutitem::
   :description: Check utilizing Fabric in interactive notebooks even with multi-GOU support.
   :header: Fabric inside notebooks
   :button_link:  fundamentals/notebooks.html

.. customcalloutitem::
   :description: Learn scaling your training on multiple devices with distributed training.
   :header: Launch distributed training
   :button_link:  fundamentals/launch.html

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

    self
    Install <fundamentals/installation>


.. toctree::
    :maxdepth: 1
    :caption: Get started in steps

    Basic skills <levels/basic>
    Intermediate skills <levels/intermediate>
    Advanced skills <levels/advanced>


.. toctree::
    :maxdepth: 1
    :caption: Core API Reference

    Fabric Arguments <api/fabric_args>
    Fabric Methods <api/fabric_methods>


.. toctree::
    :maxdepth: 1
    :caption: Full API Reference

    Accelerators <api/accelerators>
    Collectives <api/collectives>
    Environments <api/environments>
    Fabric <api/fabric>
    IO <api/io>
    Loggers <api/loggers>
    Precision <api/precision>
    Strategies <api/strategies>


.. toctree::
    :maxdepth: 1
    :name: more
    :caption: More

    Examples <examples/index>
    Glossary <glossary/index>
    How-tos <guide/index>
    Style Guide <fundamentals/code_structure>


.. raw:: html

    </div>
