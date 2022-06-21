.. _cache_calls:

#################
Caching Work Runs
#################

**Audience:** Users who want to know how ``LightningWork`` works.

**Level:** Basic

**Prerequisite**: Read the :ref:`event_loop` guide.

----

**********************************************************
What does it mean to cache the calls of Work's run method?
**********************************************************

By default, the run method in a LightningWork "remembers" (caches) the input arguments it is getting called with and does not execute again if called with the same arguments again.
In other words, the run method only executes when the input arguments have never been seen before.

This behavior can be toggled on or off:

.. code-block:: python

    # Run only when the input arguments change (default)
    work = MyWork(cache_calls=True)

    # Run everytime regardless of whether input arguments change or not
    work = MyWork(cache_calls=False)


To better understand this, imagine you want every day to sequentially download and process some data and then train a model on those data.
As explained in the pre-requisite, the Lightning App runs within an infinite while loop, so the pseudo-code of your application might looks like this:

.. code-block:: python

    from datetime import datetime

    # Lightning code
    while True:  # This is the Lightning Event Loop

        # Your code
        today = datetime.now().strftime("%D")  # '05/25/22'
        data_processor.run(today)
        train_model.run(data_processor.data)

In this scenario, you want your components to run ``once`` a day, no more! But your code is running within an infinite loop, how can this even work?
This is where the work's internal caching mechanism comes in. By default, Lightning caches a hash of the input provided to its run method and won't re-execute the method if the same input is provided again.
In the example above, the **data_processor** component run method receives the string **"05/25/22"**. It runs one time and any further execution during the day is skipped until tomorrow is reached and the work run method receives **06/25/22**. This logic applies everyday.
This caching mechanism is inspired from how `React.js Components and Props <https://reactjs.org/docs/components-and-props.html>`_ renders website. Only changes to the inputs re-trigger execution.

*******************************
How can I verify this behavior?
*******************************

Here's an example of this behavior with LightningWork:

.. literalinclude:: ../code_samples/basics/0.py
    :language: python
    :emphasize-lines: 10, 19

And you should see the following by running the code above:

.. code-block:: console

    $ python example.py
      INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
      # After you have clicked `run` on the UI.
      I received the following props: args: () kwargs: {'value': 1}
      I received the following props: args: () kwargs: {'value': 10}

As you can see, the intermediate run didn't execute, as we would expected when ``cache_calls=True``.

************************************************
What are the implications of turnin caching off?
************************************************

By setting ``cache_calls=False``, Lightning won't cache the return value and re-execute the run method on every call.

.. literalinclude:: ../code_samples/basics/1.py
    :diff: ../code_samples/basics/0.py

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

    from lightning_app import LightningApp, LightningFlow, LightningWork


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
