:orphan:

.. _install_beginner:

#############################
What is a virtual environment
#############################
A virtual environment keeps the packages you install isolated from the rest of your system.
This allows you to work on multiple projects that have different and potentially conflicting requirements, and it
keeps your system Python installation clean.

.. raw:: html

    <iframe width="560" height="315" src="https://www.youtube.com/embed/WHWsABk4Ejk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

----

We will describe two choices here, pick one:


1. :ref:`Python virtualenv <python-virtualenv>`.
2. :ref:`Conda virtual environment <conda>`.

----

.. _python-virtualenv:

********************
1. Python Virtualenv
********************

First, make sure that you have Python 3.8+ installed on your system.

.. code-block:: bash

    python3 --version

If you can't run the command above or it returns a version older than 3.8,
`install the latest version of Python <https://www.python.org/downloads/>`_.
After installing it, make sure you can run the above command without errors.

----

Creating a Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When starting with a new Python project, you typically want to create a new Python virtual environment.
Navigate to the location of your project and run the following command:

.. code-block:: bash

    python3 -m venv lightning

The name of the environment here is *lightning* but you can choose any other name you like.
By running the above command, Python will create a new folder *lightning* in the current working directory.

----

Activating the Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before you can install packages into the environment, you need to activate it:

.. code-block:: bash

    source lightning/bin/activate

You need to do this step every time you want to work on your project / open the terminal.
With your virtual environment activated, you are now ready to
:doc:`install Lightning <installation>` and get started with Apps!

----

.. _conda:

********
2. Conda
********

To get started, you first need to download and install the `Miniconda package manager <https://docs.conda.io/en/latest/miniconda.html>`_.
To check that the installation was successful, open an new terminal and run:

.. code:: bash

    conda

It should return a list of commands.

----

Creating a Conda Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When starting with a new Python project, you typically want to create a new Conda virtual environment.
Navigate to the location of your project and run the following command:

.. code-block:: bash

    conda create --yes --name lightning python=3.8

The name of the environment here is *lightning* but you can choose any other name you like.
Note how we can also specify the Python version here.

----

Activating the Conda Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before you can install packages into the environment, you need to activate it:

.. code-block:: bash

    conda activate lightning

You need to do this step every time you want to work on your project / open the terminal.
With your virtual environment activated, you are now ready to
:doc:`install Lightning <installation>` and get started with Apps!
