.. _grid:

################
AWS/GCP training
################
Lightning has a native solution for training on AWS/GCP at scale.
Go to `grid.ai <https://www.grid.ai/>`_ to create an account.

We've designed Grid to work for Lightning users without needing to make ANY changes to their code.

To use grid, take your regular command:

.. code-block:: bash

    python my_model.py --learning_rate 1e-6 --layers 2 --gpus 4

And change it to use the grid train command:

.. code-block:: bash

    grid train --grid_gpus 4 my_model.py --learning_rate 'uniform(1e-6, 1e-1, 20)' --layers '[2, 4, 8, 16]'

The above command will launch (20 * 4) experiments each running on 4 GPUs (320 GPUs!) - by making ZERO changes to
your code.

The `uniform` command is part of our new expressive syntax which lets you construct hyperparameter combinations
using over 20+ distributions, lists, etc. Of course, you can also configure all of this using yamls which
can be dynamically assembled at runtime.


.. hint:: Grid supports the search strategy of your choice! (and much more than just sweeps)
