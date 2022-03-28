.. PyTorch-Lightning documentation master file, created by
   sphinx-quickstart on Fri Nov 15 07:48:22 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyTorch Lightning
============================

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

   <hr class="docutils" style="margin: 50px 0 50px 0">


Install Lightning
-----------------


.. raw:: html

   <div class="row" style='font-size: 14px'>
      <div class='col-md-6'>

For pip (and conda) users

.. code-block:: bash

    pip install pytorch-lightning

.. raw:: html

      </div>
      <div class='col-md-6'>

Or directly from conda

.. code-block:: bash

    conda install pytorch-lightning -c conda-forge

.. raw:: html

      </div>
   </div>

Or read the `advanced install guide <starter/installation.html>`_

.. raw:: html

   <hr class="docutils" style="margin: 50px 0 50px 0">

.. raw:: html

    <div class="tutorials-callout-container">
        <div class="row">

.. Add callout items below this line

.. customcalloutitem::
   :description: Use this 2-step guide to learn the key concepts.
   :header: New to Lightning?
   :button_link:  starter/introduction.html


.. customcalloutitem::
   :description: Easily organize your existing PyTorch code as PyTorch Lightning in a few hours.
   :header: PyTorch to PyTorch Lightning
   :button_link: starter/converting.html


.. customcalloutitem::
   :description: From NLP, Computer vision to RL and meta learning - see how to use Lightning in ALL research areas.
   :header: Examples
   :button_link: tutorials.html


.. customcalloutitem::
   :description: Detailed description of API each package. Assumes you already have basic Lightning knowledge.
   :header: API Reference
   :button_link: api_references.html


.. customcalloutitem::
   :description: From hyperparameters sweeps to cloud training to Pruning and Quantization - Lightning covers the key use-cases.
   :header: Common usecases
   :button_link: common_usecases.html


.. customcalloutitem::
   :description: Learn how to benchmark PyTorch Lightning.
   :header: Benchmarking
   :button_link: benchmarking/benchmarks.html


.. raw:: html

        </div>
    </div>

.. End of callout item section

.. raw:: html

   <div style="display:none">

.. toctree::
   :maxdepth: 1
   :name: start
   :caption: Getting started

   starter/introduction
   starter/installation
   starter/converting
   starter/lightning_lite

.. toctree::
   :maxdepth: 1
   :name: guides
   :caption: Best practices

   guides/speed
   guides/data
   starter/style_guide
   Lightning project template<https://github.com/PyTorchLightning/pytorch-lightning-conference-seed>
   benchmarking/benchmarks

.. toctree::
   :maxdepth: 2
   :name: pl_docs
   :caption: Lightning API

   common/lightning_module
   common/trainer

.. toctree::
   :maxdepth: 2
   :name: docs
   :caption: Optional Extensions

   extensions/accelerator
   extensions/callbacks
   extensions/datamodules
   extensions/logging
   extensions/plugins
   extensions/strategy
   extensions/loops

.. toctree::
   :maxdepth: 1
   :name: Accelerators
   :caption: Accelerators

   accelerators/gpu
   accelerators/tpu
   accelerators/ipu
   accelerators/hpu

.. toctree::
   :maxdepth: 1
   :name: Common Use Cases
   :caption: Common Use Cases

   clouds/cloud_training
   common/checkpointing
   clouds/cluster
   common/debugging
   common/early_stopping
   advanced/training_tricks
   common/evaluation
   advanced/fault_tolerant_training
   common/hyperparameters
   common/production_inference
   common/lightning_cli
   common/loggers
   advanced/model_parallel
   advanced/precision
   common/optimization
   advanced/profiler
   common/progress_bar
   advanced/pruning_quantization
   common/remote_fs
   advanced/strategy_registry
   advanced/transfer_learning

.. toctree::
   :maxdepth: 1
   :name: Tutorials
   :caption: Tutorials
   :glob:

   starter/core_guide
   PyTorch Lightning 101 class <https://www.youtube.com/playlist?list=PLaMu-SDt_RB5NUm67hU2pdE75j6KaIOv2>
   From PyTorch to PyTorch Lightning [Blog] <https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09>
   From PyTorch to PyTorch Lightning [Video] <https://www.youtube.com/watch?v=QHww1JH7IDU>
   notebooks/**/*

.. toctree::
   :maxdepth: 2
   :name: api
   :caption: API References

   api_references

.. toctree::
   :maxdepth: 1
   :name: Lightning Ecosystem
   :caption: Lightning Ecosystem

   ecosystem/metrics
   ecosystem/flash
   ecosystem/bolts
   ecosystem/transformers
   ecosystem/ecosystem-ci

.. toctree::
   :maxdepth: 1
   :name: Examples
   :caption: Examples

   ecosystem/community_examples
   ecosystem/asr_nlp_tts
   Autoencoder <https://lightning-bolts.readthedocs.io/en/stable/deprecated/models/autoencoders.html>
   BYOL <https://lightning-bolts.readthedocs.io/en/stable/deprecated/callbacks/self_supervised.html#byolmaweightupdate>
   DQN <https://lightning-bolts.readthedocs.io/en/stable/deprecated/models/reinforce_learn.html#deep-q-network-dqn>
   GAN <https://lightning-bolts.readthedocs.io/en/stable/deprecated/models/gans.html#basic-gan>
   GPT-2 <https://lightning-bolts.readthedocs.io/en/stable/deprecated/models/convolutional.html#gpt-2>
   Image-GPT <https://lightning-bolts.readthedocs.io/en/stable/deprecated/models/convolutional.html#image-gpt>
   SimCLR <https://lightning-bolts.readthedocs.io/en/stable/deprecated/transforms/self_supervised.html#simclr-transforms>
   VAE <https://lightning-bolts.readthedocs.io/en/stable/deprecated/models/autoencoders.html#basic-vae>

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

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
