:orphan:

.. _grid_cloud_run_intermediate:

#################################
Train on the cloud (intermediate)
#################################
**Audience**: User looking to run many models at once

----

****************
What is a sweep?
****************
A sweep is the term giving to running the same model multiple times with different hyperparameters to find the one that performs the best (according to your definition of performance).

Let's say I have a python script that trains a Lighting model to classify images. We run this file like so:

.. code:: bash

      grid run file.py --batch_size 8

with such a model, I would be interested in knowing how it performs with different batch size. In this case, I'm going to train many versions of this model.

.. code:: bash

      # run 4 models in parallel
      grid run file.py --batch_size 8
      grid run file.py --batch_size 16
      grid run file.py --batch_size 32
      grid run file.py --batch_size 64

Now I can see how my model performs according to the layers and based on time and cost I can pick my "best" model:

.. list-table:: Training speed vs cost
   :widths: 10 40 15 15
   :header-rows: 1

   * - Batch size
     - classification accuracy (%)
     - training time
     - cost
   * - 8
     - 0.80
     - 5 minutes
     - $0.15
   * - 16
     - 0.85
     - 10 minutes
     - $0.30
   * - 32
     - 0.90
     - 30 minutes
     - $0.50
   * - 64
     - 0.95
     - 60 minutes
     - $1.01

----

*************
Start a Sweep
*************
First, recall that in the `previous tutorial <run_basic.rst>`_ we ran a single model using this command:

.. code:: bash

    grid run --datastore_name cifar5 cifar5.py --data_dir /datastores/cifar5

Now we're going to run that same model 4 different times each with a different number of layers:

.. code:: bash

    grid run --datastore_name cifar5 cifar5.py --data_dir /datastores/cifar5 --batch_size 8
    grid run --datastore_name cifar5 cifar5.py --data_dir /datastores/cifar5 --batch_size 16
    grid run --datastore_name cifar5 cifar5.py --data_dir /datastores/cifar5 --batch_size 32
    grid run --datastore_name cifar5 cifar5.py --data_dir /datastores/cifar5 --batch_size 64

Grid has a special syntax based on python that gives you shortcuts for sweeps. The shortcut for the above commands is:

.. code:: bash

    grid run --datastore_name cifar5 cifar5.py --data_dir /datastores/cifar5 --batch_size "[8, 16, 32, 64]"

----

****************
Syntax Shortcuts
****************

List
====

.. code:: bash

    grid run file.py --batch_size "[8, 16, 32, 64]"

equivalent to:

.. code:: bash

    grid run file.py --batch_size 8
    grid run file.py --batch_size 16
    grid run file.py --batch_size 32
    grid run file.py --batch_size 64

----

Range
=====

.. code:: bash

    grid run file.py --batch_size "range(1, 10, 2)"

equivalent to:

.. code:: bash

  grid run main.py --batch_size 1
  grid run main.py --batch_size 3
  grid run main.py --batch_size 5
  grid run main.py --batch_size 7
  grid run main.py --batch_size 9

---

String list
===========

.. code:: bash

    grid run file.py --model_backbone "['resnet18' 'transformer', 'resnet50']"

equivalent to:

.. code:: bash

  grid run file.py --model_backbone 'resnet18'
  grid run file.py --model_backbone 'transformer'
  grid run file.py --model_backbone 'resnet50'

----

Sampling
========

.. code:: bash

    grid run file.py --learning_rate "uniform(1e-5, 1e-1, 3)"

equivalent to:

.. code:: bash

    grid run file.py --learning_rate 0.03977392
    grid run file.py --learning_rate 0.04835479
    grid run file.py --learning_rate 0.05200016

----

****************
Sweep strategies
****************
Models often have dozens of hyperparameters. We usually don't run all combinations because it would be too prohibitive. Grid supports two strategies:

----

Grid search
===========
Grid search is a common approach that tries all combinations of hyperparamaters. Grid will automatically compute combinations when it detects special syntax:

.. code:: bash

    grid run file.py --batch_size "[1, 2]" --layers "[3, 5]"

is equivalent to:

.. code:: bash

    grid run file.py --batch_size 1 --layers 3
    grid run file.py --batch_size 2 --layers 3
    grid run file.py --batch_size 1 --layers 5
    grid run file.py --batch_size 2 --layers 5

----

Random search
=============
With random search, we choose only a subset of hyperparamaters. The larger the number of trials (*num_trials*) the more probable we'll find a great performing model without needing to try all possible combinations.

.. code:: bash

    grid run --strategy random_search --num_trials 2 file.py --batch_size "[1, 2]" --layers "[3, 5]"

the above command generates the 4 combinations and runs only 2 at random

.. code:: bash

    grid run file.py --batch_size 2 --layers 3
    grid run file.py --batch_size 1 --layers 5

----

**********
Next Steps
**********
Here are the recommended next steps depending on your workflow.

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Run with your own cloud credentials
   :description: Learn how to use Grid products with your Company or University cloud account.
   :col_css: col-md-4
   :button_link: run_expert.html
   :height: 180
   :tag: expert

.. raw:: html

        </div>
    </div
