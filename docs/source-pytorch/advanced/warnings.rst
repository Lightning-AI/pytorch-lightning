########
Warnings
########

Lightning warns users of possible misconfiguration, performance implications or potential mistakes through the ``PossibleUserWarning`` category.
Sometimes these warnings can be false positives, and you may want to suppress them to avoid cluttering the logs.


.. warning::

    Suppressing warnings is not recommended in general, because they may raise important issues that you should address.
    Only suppress warnings if they are false.


-----


*********************************
Suppress a single warning message
*********************************

Suppressing an individual warning message can be done through the :mod:`warnings` module:

.. code-block:: python

    import warnings

    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")


-----


*********************************************
Suppress all instances of PossibleUserWarning
*********************************************

Suppressing all warnings of the ``PossibleUserWarning`` category can be done programmatically

.. code-block:: python

    from lightning.pytorch.utilities import disable_possible_user_warnings

    # ignore all warnings that could be false positives
    disable_possible_user_warnings()

or through the environment variable ``POSSIBLE_USER_WARNINGS``:


.. code-block:: bash

    export POSSIBLE_USER_WARNINGS=off
    # or
    export POSSIBLE_USER_WARNINGS=0
