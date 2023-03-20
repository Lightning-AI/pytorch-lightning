:orphan:

.. _build_model:

#######################
Build and Train a Model
#######################

**Required background:** Basic Python familiarity and complete the  guide.

**Goal:** We'll walk you through the creation of a model using PyTorch Lightning.

----

*********************************
A simple PyTorch Lightning script
*********************************

Let's assume you already have a folder with those two files.

.. code-block:: bash

    pl_project/
        train.py            # your own script to train your models
        requirements.txt    # your python requirements.

If you don't, simply create a ``pl_project`` folder with those two files and add the following `PyTorch Lightning <https://lightning.ai/docs/pytorch/latest/>`_ code  in the ``train.py`` file. This code trains a simple ``AutoEncoder`` on `MNIST Dataset <https://en.wikipedia.org/wiki/MNIST_database>`_.

.. literalinclude:: ../code_samples/convert_pl_to_app/train.py

Add the following to the ``requirements.txt`` file.

.. literalinclude:: ../code_samples/convert_pl_to_app/requirements.txt

Simply run the following commands in your terminal to install the requirements and train the model.

.. code-block:: bash

    pip install -r requirements.txt
    python train.py

Get through `PyTorch Lightning Introduction <https://lightning.ai/docs/pytorch/stable/starter/introduction.html#step-1-define-lightningmodule>`_ to learn more.

----

**********
Next Steps
**********

.. raw:: html

   <br />
   <div class="display-card-container">
      <div class="row">

.. displayitem::
   :header: Evolve a Model into an ML System
   :description: Develop an App to train a model in the cloud
   :col_css: col-md-6
   :button_link: training_with_apps.html
   :height: 180

.. displayitem::
   :header: Start from a Template ML System
   :description: Learn about Apps, from a template.
   :col_css: col-md-6
   :button_link: go_beyond_training.html
   :height: 180

.. raw:: html

      </div>
   </div>
