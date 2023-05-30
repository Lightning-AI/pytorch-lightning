:orphan:

.. _lightning-cli:

######################################
Configure hyperparameters from the CLI
######################################

*************
Why use a CLI
*************

When running deep learning experiments, there are a couple of good practices that are recommended to follow:

- Separate configuration from source code
- Guarantee reproducibility of experiments

Implementing a command line interface (CLI) makes it possible to execute an experiment from a shell terminal. By having
a CLI, there is a clear separation between the Python source code and what hyperparameters are used for a particular
experiment. If the CLI corresponds to a stable version of the code, reproducing an experiment can be achieved by
installing the same version of the code plus dependencies and running with the same configuration (CLI arguments).

----

*********
Basic use
*********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: 1: Control it all from the CLI
   :description: Learn to control a LightningModule and LightningDataModule from the CLI
   :col_css: col-md-4
   :button_link: lightning_cli_intermediate.html
   :height: 150
   :tag: intermediate

.. displayitem::
   :header: 2: Mix models, datasets and optimizers
   :description: Support multiple models, datasets, optimizers and learning rate schedulers
   :col_css: col-md-4
   :button_link: lightning_cli_intermediate_2.html
   :height: 150
   :tag: intermediate

.. displayitem::
   :header: 3: Control it all via YAML
   :description: Enable composable YAMLs
   :col_css: col-md-4
   :button_link: lightning_cli_advanced.html
   :height: 150
   :tag: advanced

.. raw:: html

        </div>
    </div>

----

************
Advanced use
************

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: YAML for production
   :description: Use the Lightning CLI with YAMLs for production environments
   :col_css: col-md-4
   :button_link: lightning_cli_advanced_2.html
   :height: 150
   :tag: advanced

.. displayitem::
   :header: Customize for complex projects
   :description: Learn how to implement CLIs for complex projects
   :col_css: col-md-4
   :button_link: lightning_cli_advanced_3.html
   :height: 150
   :tag: advanced

.. displayitem::
   :header: Extend the Lightning CLI
   :description: Customize the Lightning CLI
   :col_css: col-md-4
   :button_link: lightning_cli_expert.html
   :height: 150
   :tag: expert

----

*************
Miscellaneous
*************

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: FAQ
   :description: Frequently asked questions about working with the Lightning CLI and YAML files
   :col_css: col-md-6
   :button_link: lightning_cli_faq.html
   :height: 150

.. raw:: html

        </div>
    </div>
