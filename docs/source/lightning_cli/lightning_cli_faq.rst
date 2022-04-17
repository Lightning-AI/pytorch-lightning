#####################################
Eliminate config boilerplate (expert)
#####################################

Troubleshooting
^^^^^^^^^^^^^^^

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


Notes related to reproducibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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