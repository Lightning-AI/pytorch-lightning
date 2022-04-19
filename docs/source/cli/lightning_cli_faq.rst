:orphan:

.. testsetup:: *
    :skipif: not _JSONARGPARSE_AVAILABLE

    import torch
    from unittest import mock
    from typing import List
    import pytorch_lightning as pl
    from pytorch_lightning import LightningModule, LightningDataModule, Trainer, Callback


    class NoFitTrainer(Trainer):
        def fit(self, *_, **__):
            pass


    class LightningCLI(pl.utilities.cli.LightningCLI):
        def __init__(self, *args, trainer_class=NoFitTrainer, run=False, **kwargs):
            super().__init__(*args, trainer_class=trainer_class, run=run, **kwargs)


    class MyModel(LightningModule):
        def __init__(
            self,
            encoder_layers: int = 12,
            decoder_layers: List[int] = [2, 4],
            batch_size: int = 8,
        ):
            pass


    mock_argv = mock.patch("sys.argv", ["any.py"])
    mock_argv.start()

.. testcleanup:: *

    mock_argv.stop()

#####################################
Eliminate config boilerplate (expert)
#####################################

***************
Troubleshooting
***************
The standard behavior for CLIs, when they fail, is to terminate the process with a non-zero exit code and a short message
to hint the user about the cause. This is problematic while developing the CLI since there is no information to track
down the root of the problem. A simple change in the instantiation of the ``LightningCLI`` can be used such that when
there is a failure an exception is raised and the full stack trace printed.

.. testcode::

    cli = LightningCLI(MyModel, parser_kwargs={"error_handler": None})

.. note::

    When asking about problems and reporting issues please set the ``error_handler`` to ``None`` and include the stack
    trace in your description. With this, it is more likely for people to help out identifying the cause without needing
    to create a reproducible script.

----

*************************************
Reproducibility with the LightningCLI
*************************************
The topic of reproducibility is complex and it is impossible to guarantee reproducibility by just providing a class that
people can use in unexpected ways. Nevertheless, the :class:`~pytorch_lightning.utilities.cli.LightningCLI` tries to
give a framework and recommendations to make reproducibility simpler.

When an experiment is run, it is good practice to use a stable version of the source code, either being a released
package or at least a commit of some version controlled repository. For each run of a CLI the config file is
automatically saved including all settings. This is useful to figure out what was done for a particular run without
requiring to look at the source code. If by mistake the exact version of the source code is lost or some defaults
changed, having the full config means that most of the information is preserved.

The class is targeted at implementing CLIs because running a command from a shell provides a separation with the Python
source code. Ideally the CLI would be placed in your path as part of the installation of a stable package, instead of
running from a clone of a repository that could have uncommitted local modifications. Creating installable packages that
include CLIs is out of the scope of this document. This is mentioned only as a teaser for people who would strive for
the best practices possible.


For every CLI implemented, users are encouraged to learn how to run it by reading the documentation printed with the
:code:`--help` option and use the :code:`--print_config` option to guide the writing of config files. A few more details
that might not be clear by only reading the help are the following.

:class:`~pytorch_lightning.utilities.cli.LightningCLI` is based on argparse and as such follows the same arguments style
as many POSIX command line tools. Long options are prefixed with two dashes and its corresponding values should be
provided with an empty space or an equal sign, as :code:`--option value` or :code:`--option=value`. Command line options
are parsed from left to right, therefore if a setting appears multiple times the value most to the right will override
the previous ones. If a class has an init parameter that is required (i.e. no default value), it is given as
:code:`--option` which makes it explicit and more readable instead of relying on positional arguments.

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
        tune                Runs routines to tune hyperparameters before training.

use a subcommand as follows:

.. code:: bash

    python main.py fit
    python main.py test

----

****************
What is the CLI?
****************
CLI is short for commandline interface. Use your terminal to enter these commands.
