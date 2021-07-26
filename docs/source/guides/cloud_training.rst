.. _grid:

##############
Cloud Training
##############

Lightning has a native solution for training on AWS/GCP at scale.
Go to `grid.ai <https://www.grid.ai/>`_ to create an account.

We've designed Grid to work seamlessly with Lightning, without needing to make ANY code changes.

To use Grid, replace ``python`` in your regular command:

.. code-block:: bash

    python my_model.py --learning_rate 1e-6 --layers 2 --gpus 4

To use the ``grid train`` command:

.. code-block:: bash

    grid train --grid_gpus 4 my_model.py --learning_rate 'uniform(1e-6, 1e-1, 20)' --layers '[2, 4, 8, 16]'

The above command will launch (20 * 4) experiments, each running on 4 GPUs (320 GPUs!) - by making ZERO changes to
your code.

The ``uniform`` command is part of our new expressive syntax which lets you construct hyperparameter combinations
using over 20+ distributions, lists, etc. Of course, you can also configure all of this using yamls which
can be dynamically assembled at runtime.

***************
Grid Highlights
***************

* Run any public or private repository with Grid, or use an interactive session.
* Grid allocates all the machines and GPUs you need on demand, so you only pay for what you need when you need it.
* Grid handles all the other parts of developing and training at scale: artifacts, logs, metrics, etc.
* Grid works with the experiment manager of your choice, no code changes needed.
* Use Grid Datastores- high-performance, low-latency, versioned datasets.
