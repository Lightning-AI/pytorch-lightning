
********************************************************
What caching the calls of Work's run method does for you
********************************************************

By default, the run method in a LightningWork (Work) "remembers" (caches) the input arguments it is getting called with and does not execute again if called with the same arguments again.
In other words, the run method only executes when the input arguments have never been seen before.

You can turn caching on or off:

.. code-block:: python

    # Run only when the input arguments change (default)
    work = MyWork(cache_calls=True)

    # Run every time regardless of whether input arguments change or not
    work = MyWork(cache_calls=False)

To better understand this, imagine that every day you want to sequentially download and process some data and then train a model on that data.
As explained in the :doc:`Event Loop guide <../../glossary/event_loop>`, the Lightning App runs within an infinite while loop, so the pseudo-code of your application might looks like this:

.. code-block:: python

    from datetime import datetime

    # Lightning code
    while True:  # This is the Lightning Event Loop

        # Your code
        today = datetime.now().strftime("%D")  # '05/25/22'
        data_processor.run(today)
        train_model.run(data_processor.data)

In this scenario, you want your components to run ``once`` a day, and no more than that! But your code is running within an infinite loop, how can this even work?
This is where the Work's internal caching mechanism comes in. By default, Lightning caches a hash of the input provided to its run method and won't re-execute the method if the same input is provided again.
In the example above, the **data_processor** component run method receives the string **"05/25/22"**. It runs one time and any further execution during the day is skipped until tomorrow is reached and the work run method receives **06/25/22**. This logic applies everyday.
This caching mechanism is inspired from how `React.js Components and Props <https://reactjs.org/docs/components-and-props.html>`_ renders websites. Only changes to the inputs re-trigger execution.

***************
Caching Example
***************

Here's an example of this behavior with LightningWork:

.. code:: python
    :emphasize-lines: 11, 17

    import lightning as L


    class ExampleWork(L.LightningWork):
        def run(self, *args, **kwargs):
            print(f"I received the following props: args: {args} kwargs: {kwargs}")


    work = ExampleWork()
    work.run(value=1)

    # Providing the same value. This won't run as already cached.
    work.run(value=1)
    work.run(value=1)
    work.run(value=1)
    work.run(value=1)

    # Changing the provided value. This isn't cached and will run again.
    work.run(value=10)

And you should see the following by running the code above:

.. code-block:: console

    $ python example.py
      INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
      # After you have clicked `run` on the UI.
      I received the following props: args: () kwargs: {'value': 1}
      I received the following props: args: () kwargs: {'value': 10}

As you can see, the intermediate run didn't execute, as we would expected when ``cache_calls=True``.

***********************************
Implications of turning caching off
***********************************

By setting ``cache_calls=False``, Lightning won't cache the return value and re-execute the run method on every call.

.. code:: python
    :emphasize-lines: 7

    from lightning.app import LightningWork


    class ExampleWork(LightningWork):
        def run(self, *args, **kwargs):
            print(f"I received the following props: args: {args} kwargs: {kwargs}")


    work = ExampleWork(cache_calls=False)
    work.run(value=1)

    # Providing the same value. This won't run as already cached.
    work.run(value=1)
    work.run(value=1)
    work.run(value=1)
    work.run(value=1)

    # Changing the provided value. This isn't cached and will run again.
    work.run(value=10)

.. code-block:: console

    $ python example.py
      INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
      # After you have clicked `run` on the UI.
      I received the following props: args: () kwargs: {'value': 1}
      I received the following props: args: () kwargs: {'value': 1}
      I received the following props: args: () kwargs: {'value': 1}
      I received the following props: args: () kwargs: {'value': 1}
      I received the following props: args: () kwargs: {'value': 1}
      I received the following props: args: () kwargs: {'value': 10}

Be aware than when setting both ``cache_calls=False`` and ``parallel=False`` to a work, the code after the ``self.work.run()`` is unreachable
as the work continuously execute in a blocking way.

.. code-block:: python
    :emphasize-lines: 9-10

    from lightning.app import LightningApp, LightningFlow, LightningWork


    class Flow(LightningFlow):
        def __init__(self):
            super().__init__()

            self.work = Work(cache_calls=False, parallel=False)

        def run(self):
            print("HERE BEFORE")
            self.work.run()
            print("HERE AFTER")


    app = LightningApp(Flow())

.. code-block:: console

    $ lightning run app app.py
      INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
      print("HERE BEFORE")
      print("HERE BEFORE")
      print("HERE BEFORE")
      ...
