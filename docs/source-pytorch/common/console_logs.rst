###############
Console logging
###############
**Audience:** Engineers looking to capture more visible logs.

----

*******************
Enable console logs
*******************
Lightning logs useful information about the training process and user warnings to the console.
You can retrieve the Lightning console logger and change it to your liking. For example, adjust the logging level
or redirect output for certain modules to log files:

.. testcode::

    import logging

    # configure logging at the root level of Lightning
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

    # configure logging on module level, redirect to file
    logger = logging.getLogger("lightning.pytorch.core")
    logger.addHandler(logging.FileHandler("core.log"))

Read more about custom Python logging `here <https://docs.python.org/3/library/logging.html>`_.
