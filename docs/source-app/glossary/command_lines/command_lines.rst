:orphan:

############################
Command Line Interface (CLI)
############################

**Audience:** Users looking to create a command line interface for their application.

----

****************************************
What is a command line interface (CLI) ?
****************************************

A command-line interface (CLI) is a text-based user interface (UI) used to run programs, manage computer files and interact with the computer. Command-line interfaces are also called command-line user interfaces, console user interfaces and character user interfaces.

As an example, Lightning provides a CLI to interact with your Lightning Apps and the `lightning.ai <https://lightning.ai/>`_ platform as follows:

.. code-block:: bash

    main
    ├── create - Create Lightning AI self-managed resources (clusters, etc…)
    │   └── cluster - Create a Lightning AI BYOC compute cluster with your cloud provider credentials.
    ├── delete - Delete Lightning AI self-managed resources (clusters, etc…)
    │   └── cluster - Delete a Lightning AI BYOC compute cluster and all associated cloud provider resources.
    ├── fork - Fork an application.
    ├── init - Init a Lightning App and/or component.
    │   ├── app
    │   ├── component
    │   ├── pl-app - Create an app from your PyTorch Lightning source files.
    │   └── react-ui - Create a react UI to give a Lightning component a React.js web user interface (UI)
    ├── install - Install a Lightning App and/or component.
    │   ├── app
    │   └── component
    ├── list - List Lightning AI self-managed resources (clusters, etc…)
    │   ├── apps - List your Lightning AI apps.
    │   └── clusters - List your Lightning AI BYOC compute clusters.
    ├── login - Log in to your lightning.ai account.
    ├── logout - Log out of your lightning.ai account.
    ├── run - Run a Lightning application locally or on the cloud.
    │   └── app - Run an app from a file.
    ├── show - Show given resource.
    │   ├── cluster - Groups cluster commands inside show.
    │   │   └── logs - Show cluster logs.
    │   └── logs - Show cloud application logs. By default prints logs for all currently available components.
    ├── stop - Stop your application.
    └── tree - show the command tree of your CLI

Learn more with `Command-line interface <https://en.wikipedia.org/wiki/Command-line_interface>`_.

.. note::

    The Lightning guideline to build CLI is lightning <VERB> <NOUN> ... or <ACTION> <OBJECT> ...

----

**********
Learn more
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Develop a Command Line Interface
   :description: Learn how to develop an CLI for your application.
   :col_css: col-md-6
   :button_link: ../../workflows/build_command_line_interface/index_content.html
   :height: 150

.. raw:: html

        </div>
    </div>
