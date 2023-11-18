:orphan:

###########################################
Frequently asked questions for LightningCLI
###########################################

************************
What does CLI stand for?
************************
CLI is short for command line interface. This means it is a tool intended to be run from a terminal, similar to commands
like ``git``.

----

.. _what-is-a-yaml-config-file:

***************************
What is a yaml config file?
***************************
A YAML is a standard for configuration files used to describe parameters for sections of a program. It is a common tool
in engineering and has recently started to gain popularity in machine learning. An example of a YAML file is the
following:

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

If you are unfamiliar with YAML, the short introduction at `realpython.com#yaml-syntax
<https://realpython.com/python-yaml/#yaml-syntax>`__ might be a good starting point.

----

*********************
What is a subcommand?
*********************
A subcommand is what is the action the LightningCLI applies to the script:

.. code:: bash

    python main.py [subcommand]

See the Potential subcommands with:

.. code:: bash

    python main.py --help

which prints:

.. code:: bash

        ...

        fit                 Runs the full optimization routine.
        validate            Perform one evaluation epoch over the validation set.
        test                Perform one evaluation epoch over the test set.
        predict             Run inference on your data.

use a subcommand as follows:

.. code:: bash

    python main.py fit
    python main.py test

----

*******************************************************
What is the relation between LightningCLI and argparse?
*******************************************************

:class:`~lightning.pytorch.cli.LightningCLI` makes use of `jsonargparse <https://github.com/omni-us/jsonargparse>`__
which is an extension of `argparse <https://docs.python.org/3/library/argparse.html>`__. Due to this,
:class:`~lightning.pytorch.cli.LightningCLI` follows the same arguments style as many POSIX command line tools. Long
options are prefixed with two dashes and its corresponding values are separated by space or an equal sign, as ``--option
value`` or ``--option=value``. Command line options are parsed from left to right, therefore if a setting appears
multiple times, the value most to the right will override the previous ones.

----

*******************************************
What is the override order of LightningCLI?
*******************************************

The final configuration of CLIs implemented with :class:`~lightning.pytorch.cli.LightningCLI` can depend on default
config files (if defined), environment variables (if enabled) and command line arguments. The override order between
these is the following:

1. Defaults defined in the source code.
2. Existing default config files in the order defined in ``default_config_files``, e.g. ``~/.myapp.yaml``.
3. Entire config environment variable, e.g. ``PL_FIT__CONFIG``.
4. Individual argument environment variables, e.g. ``PL_FIT__SEED_EVERYTHING``.
5. Command line arguments in order left to right (might include config files).

----

****************************
How do I troubleshoot a CLI?
****************************
The standard behavior for CLIs, when they fail, is to terminate the process with a non-zero exit code and a short
message to hint the user about the cause. This is problematic while developing the CLI since there is no information to
track down the root of the problem. To troubleshoot set the environment variable ``JSONARGPARSE_DEBUG`` to any value
before running the CLI:

.. code:: bash

    export JSONARGPARSE_DEBUG=true
    python main.py fit

.. note::

    When asking about problems and reporting issues, please set the ``JSONARGPARSE_DEBUG`` and include the stack trace
    in your description. With this, users are more likely to help identify the cause without needing to create a
    reproducible script.
