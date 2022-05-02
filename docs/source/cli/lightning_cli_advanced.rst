:orphan:

#######################################
Eliminate config boilerplate (Advanced)
#######################################
**Audience:** Users looking to modularize their code for a professional project.

**Pre-reqs:** You must have read :doc:`(Control it all from the CLI) <lightning_cli_intermediate>`.

----

***************************
What is a yaml config file?
***************************
A yaml is a standard configuration file that describes parameters for sections of a program. It is a common tool in engineering, and it has recently started to gain popularity in machine learning.

.. code:: yaml

    # file.yaml
    car:
        max_speed:100
        max_passengers:2
    plane:
        fuel_capacity: 50
    class_3:
        option_1: 'x'
        option_2: 'y'

----


*********************
Print the config used
*********************
Before or after you run a training routine, you can print the full training spec in yaml format using ``--print_config``:

.. code:: bash

    python main.py fit --print_config

which generates the following config:

.. code:: bash

    seed_everything: null
    trainer:
        logger: true
        ...
        terminate_on_nan: null
    model:
        out_dim: 10
        learning_rate: 0.02
    data:
        data_dir: ./
    ckpt_path: null

----

********************************
Write a config yaml from the CLI
********************************
To have a copy of the configuration that produced this model, save a *yaml* file from the *--print_config* outputs:

.. code:: bash

    python main.py fit --model.learning_rate 0.001 --print_config > config.yaml

----

**********************
Run from a single yaml
**********************
To run from a yaml, pass a yaml produced with ``--print_config`` to the ``--config`` argument:

.. code:: bash

    python main.py fit --config config.yaml

when using a yaml to run, you can still pass in inline arguments

.. code:: bash

    python main.py fit --config config.yaml --trainer.max_epochs 100

----

******************
Compose yaml files
******************
For production or complex research projects it's advisable to have each object in its own config file. To compose all the configs, pass them all inline:

.. code-block:: bash

    $ python trainer.py fit --config trainer.yaml --config datamodules.yaml --config models.yaml ...

The configs will be parsed sequentially. Let's say we have two configs with the same args:

.. code:: yaml

    # trainer.yaml
    trainer:
        num_epochs: 10


    # trainer_2.yaml
    trainer:
        num_epochs: 20

the ones from the last config will be used (num_epochs = 20) in this case:

.. code-block:: bash

    $ python trainer.py fit --config trainer.yaml --config trainer_2.yaml
