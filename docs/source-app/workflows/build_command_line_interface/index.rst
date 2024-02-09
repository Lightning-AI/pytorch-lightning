############################
Command-line Interface (CLI)
############################

**Audience:** Users looking to create a command line interface (CLI) for their application.

----

**************
What is a CLI?
**************

A Command-line Interface (CLI) is an user interface (UI) in a terminal to interact with a specific program.

.. note::

    The Lightning guideline to build CLI is `lightning_app <VERB> <NOUN> ...` or `<ACTION> <OBJECT> ...`.

As an example, Lightning provides a CLI to interact with your Lightning Apps and the `lightning.ai <https://lightning.ai/>`_ platform as follows:

.. code-block:: bash

    main
    ├── fork - Forks an App.
    ├── init - Initializes a Lightning App and/or Component.
    │   ├── app
    │   ├── component
    │   ├── pl-app - Creates an App from your PyTorch Lightning source files.
    │   └── react-ui - Creates a React UI to give a Lightning Component a React.js web UI
    ├── install - Installs a Lightning App and/or Component.
    │   ├── app
    │   └── component
    ├── list - Lists Lightning AI self-managed resources (apps)
    │   └── apps - Lists your Lightning AI Apps.
    ├── login - Logs in to your lightning.ai account.
    ├── logout - Logs out of your lightning.ai account.
    ├── run - Runs a Lightning App locally or on the cloud.
    │   └── app - Runs an App from a file.
    ├── show - Shows given resource.
    │   └── logs - Shows cloud application logs. By default prints logs for all currently available Components.
    ├── stop - Stops your App.
    └── tree - Shows the command tree of your CLI.

Learn more about `Command-line interfaces here <https://en.wikipedia.org/wiki/Command-line_interface>`_.

----

.. include:: index_content.rst
