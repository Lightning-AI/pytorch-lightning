.. PyTorch-Lightning documentation master file, created by
   sphinx-quickstart on Fri Nov 15 07:48:22 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyTorch Lightning
============================

TODO: write why lightning is not


----


Install Lightning 
-----------------

For pip users (tip: pip also works inside conda environments)

.. code-block:: bash

    pip install pytorch-lightning


For conda (miniconda, anaconda) users

.. code-block:: bash

    conda install pytorch-lightning -c conda-forge

Or read the `advanced install guide <starter/installation.html>`_

----

.. raw:: html

    <div class="tutorials-callout-container">
        <div class="row">

.. Add callout items below this line

.. customcalloutitem::
   :description: Use this 2-step guide to learn the key concepts.
   :header: New to Lightning?
   :button_link:  starter/introduction.html
   :button_text: Continue 

.. customcalloutitem::
   :description: Learn the key concepts to help you organize your PyTorch code as PyTorch Lightning
   :header: PyTorch to PyTorch Lightning
   :button_link: starter/converting.html
   :button_text: Continue

.. customcalloutitem::
   :description: Learn how to use Lightning for anything from NLP, Computer vision to RL and meta learning. Lightning works for ALL research areas.
   :header: Examples
   :button_link: tutorials.html
   :button_text: Continue

.. customcalloutitem::
   :description: Detailed description of API each package. Assumes you already have basic Lightning knowledge.
   :header: API Reference
   :button_link: api_references.html
   :button_text: Continue

.. customcalloutitem::
   :description: Learn how to do anything from hyperparameters sweeps to cloud training to Pruning and Quantization.
   :header: Common usecases
   :button_link: usecases.html
   :button_text: Continue

.. customcalloutitem::
   :description: Learn how we benchmark against non-organized PyTorch
   :header: Benchmarking
   :button_link: benchmarking/benchmarks.html
   :button_text: Continue

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
   clouds/cluster
   common/debugging
   common/early_stopping
   common/hyperparameters
   common/production_inference
   common/lightning_cli
   common/loggers
   advanced/model_parallel
   advanced/precision
   common/checkpointing
   advanced/fault_tolerant_training
   common/optimization
   advanced/profiler
   advanced/strategy_registry
   common/remote_fs
   advanced/training_tricks
   advanced/pruning_quantization
   common/progress_bar
   advanced/transfer_learning
   common/evaluation

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
