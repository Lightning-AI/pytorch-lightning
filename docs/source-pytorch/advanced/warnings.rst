########
Warnings
########

Lightning warns users of possible misconfiguration, performance implications or potential mistakes through the :class:`~lightning.fabric.utilities.warnings.PossibleUserWarning` category.
Sometimes these warnings can be false positives, and you may want to suppress them to avoid cluttering the logs.


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

Suppressing all warnings of the :class:`~lightning.fabric.utilities.warnings.PossibleUserWarning` category can be done programmatically

.. code-block:: python

    import lightning as L

    # ignore all warnings that could be false positives
    L.disable_possible_user_warnings()

or through the environment variable ``POSSIBLE_USER_WARNINGS``:


.. code-block:: bash

    export POSSIBLE_USER_WARNINGS=off


.. warning::

    Suppressing warnings is not recommended, as it may hide important performance issues or misconfigurations.
    Proceed with caution.
