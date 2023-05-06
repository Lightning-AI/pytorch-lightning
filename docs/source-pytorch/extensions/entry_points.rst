************
Entry Points
************

Lightning supports registering Trainer callbacks directly through
`Entry Points <https://setuptools.pypa.io/en/latest/userguide/entry_point.html>`_. Entry points allow an arbitrary
package to include callbacks that the Lightning Trainer can automatically use, without you having to add them
to the Trainer manually. This is useful in production environments where it is common to provide specialized monitoring
and logging callbacks globally for every application.

Here is a callback factory function that returns two special callbacks:

.. code-block:: python
    :caption: factories.py

    def my_custom_callbacks_factory():
        return [MyCallback1(), MyCallback2()]

If we make this `factories.py` file into an installable package, we can define an **entry point** for this factory function.
Here is a minimal example of the `setup.py` file for the package `my-package`:

.. code-block:: python
    :caption: setup.py

    from setuptools import setup

    setup(
        name="my-package",
        version="0.0.1",
        install_requires=["lightning"],
        entry_points={
            "lightning.pytorch.callbacks_factory": [
                # The format here must be [any name]=[module path]:[function name]
                "monitor_callbacks=factories:my_custom_callbacks_factory"
            ]
        },
    )

The group name for the entry points is ``lightning.pytorch.callbacks_factory`` and it contains a list of strings that
specify where to find the function within the package.

Now, if you `pip install -e .` this package, it will register the ``my_custom_callbacks_factory`` function and Lightning
will automatically call it to collect the callbacks whenever you run the Trainer!

To unregister the factory, simply uninstall the package with `pip uninstall "my-package"`.
