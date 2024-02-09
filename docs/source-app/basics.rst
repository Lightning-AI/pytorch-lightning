:orphan:

.. _basics:

######
Basics
######

In this guide, we'll cover the basic terminology associated with the Lightning framework.

----

**************
Lightning App
**************

The :class:`~lightning.app.core.app.LightningApp` runs a tree of one or more components that interact to create end-to-end applications. There are two kinds of components: :class:`~lightning.app.core.flow.LightningFlow` and :class:`~lightning.app.core.work.LightningWork`. This modular design enables you to reuse components created by other users.

----

Lightning Work
^^^^^^^^^^^^^^

The :class:`~lightning.app.core.work.LightningWork` component is a building block optimized for long-running jobs or integrating third-party services. LightningWork can be used for training large models, downloading a dataset, or any long-lasting operation.

----

Lightning Flow
^^^^^^^^^^^^^^

The :class:`~lightning.app.core.flow.LightningFlow` component coordinates long-running tasks :class:`~lightning.app.core.work.LightningWork` and runs its children :class:`~lightning.app.core.flow.LightningFlow` components.

----

Lightning App Tree
^^^^^^^^^^^^^^^^^^

Components can be nested to form component trees where the LightningFlows are its branches and LightningWorks are its leaves.

Here's a basic application with four flows and two works:

.. literalinclude:: code_samples/quickstart/app_comp.py

And here's its associated tree structure:

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/tree.gif
    :alt: Basic App Components
    :width: 100 %

A Lightning App runs all flows into a single process. Its flows coordinate the execution of the works each running in their own independent processes.

----

Lightning Distributed Event Loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Drawing inspiration from modern web frameworks like `React.js <https://reactjs.org/>`_, the Lightning app runs all flows in an **event loop** (forever), which is triggered every 0.1 seconds after collecting any works' state change.

.. figure::  https://pl-public-data.s3.amazonaws.com/assets_lightning/lightning_loop.gif

When running an app in the cloud, the :class:`~lightning.app.core.work.LightningWork` run on different machines. Lightning communicates any :class:`~lightning.app.core.work.LightningWork` state changes to the **event loop** which re-executes the flow with the newly-collected works' state.

----

Lightning App State
^^^^^^^^^^^^^^^^^^^

By design, each component is stateful and its state is composed of all its attributes. The **Lightning App State** is the collection of all its components state.

With this mechanism, any component can **react** to any other component **state changes**, simply by relying on its attributes within the flow.

For example, here we define two flow components, **RootFlow** and **ChildFlow**, where the child flow prints and increments a counter indefinitely and gets reflected in **RootFlow** state.

You can easily check the state of your entire app:

.. literalinclude:: code_samples/quickstart/app_01.py

Here's the entire tree structure associated with your app:

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/parent_child.png
    :alt: Parent Child Components
    :width: 100 %

And here's the output you get when running the above application using **Lightning CLI**:

.. code-block:: console

    $ lightning_app run app docs/source/code_samples/quickstart/app_01.py
      INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
      State: {'works': {'w_1': {'vars': {'counter': 1}}, 'w_2': {'vars': {'counter': 0}}}}

      State: {'works': {'w_1': {'vars': {'counter': 3}}, 'w_2': {'vars': {'counter': 1}}}}

      State: {'works': {'w_1': {'vars': {'counter': 4}}, 'w_2': {'vars': {'counter': 1}}}}

      State: {'works': {'w_1': {'vars': {'counter': 5}}, 'w_2': {'vars': {'counter': 2}}}}

      State: {'works': {'w_1': {'vars': {'counter': 6}}, 'w_2': {'vars': {'counter': 2}}}}

      State: {'works': {'w_1': {'vars': {'counter': 7}}, 'w_2': {'vars': {'counter': 3}}}}
      ...

This app will count forever because the **lightning event loop** indefinitely calls the root flow run method.

----

*******************************
Controlling the Execution Flow
*******************************


LightningWork: To Cache or Not to Cache Calls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With Lightning, you can control how to run your components.

By default, the :class:`~lightning.app.core.flow.LightningFlow` is executed infinitely by the **Lightning Infinite Loop** and the :class:`~lightning.app.core.work.LightningWork` does not run in **parallel**,
meaning the **Lightning Infinite Loop** (a.k.a the flow) waits until that long-running work is completed to continue.

Similar to `React.js Components and Props <https://reactjs.org/docs/components-and-props.html>`_, the :class:`~lightning.app.core.work.LightningWork`
component accepts arbitrary inputs (the "props") to its **run** method and by default runs **once** for each unique input provided.

Here's an example of this behavior:

.. literalinclude:: code_samples/basics/0.py
    :language: python
    :emphasize-lines: 10, 19

And you should see the following by running the code above:

.. code-block:: console

    $ python example.py
      INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
      # After you have clicked `run` on the UI.
      I received the following props: args: () kwargs: {'value': 1}
      I received the following props: args: () kwargs: {'value': 10}

As you can see, the intermediate run didn't execute as already cached.

To disable this behavior, set ``cache_calls=False`` to make any LightningWork run infinitely.

.. literalinclude:: code_samples/basics/1.py
    :diff: code_samples/basics/0.py

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


.. note:: Passing a sequence of different props to the work run method queues their execution. We recommend avoiding this behavior as it can be hard to debug. Instead, wait for the previous run to execute.

----

LightningWork: Parallel vs Non Parallel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The LightningWork component is made for long-running jobs.

As an example, let's create a long-running **LightningWork** component that will take 1 hour to do its "work".

.. literalinclude:: code_samples/quickstart/app_02.py
    :language: python
    :emphasize-lines: 15

Here's the output you get when running the above application using **Lightning CLI**:

.. code-block:: console

    $ lightning_app run app docs/source/code_samples/quickstart/app_02.py
      INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
      # After you have clicked `run` on the UI.
      0.0 0.0
      ...
      0.0003 0.0003
      ...
      1.0 1.0
      ...
      1 hour later!
      1.0 1.0
      1 hour later!
      1.0 1.0
      1 hour later!
      ...

The child work runs only once, hence why the progress counter stops increasing once the work is completed.

This is useful for monitoring the progress of a long-running operation, like training a big model.

.. note ::
    The Lightning Infinite Loop runs multiple cycles per second.
    It is good practice to keep the loop running fast, so that your application stays responsive,
    especially when it contains user-interface components.

----

****************
Multiple works
****************

In practical use cases, you might want to execute multiple long-running works in parallel.

To enable this behavior, set ``parallel=True`` in the ``__init__`` method of
your :class:`~lightning.app.core.work.LightningWork`.

Here's an example of the interaction between parallel and non-parallel behaviors:

.. figure:: https://pl-public-data.s3.amazonaws.com/assets_lightning/blocking_non_blocking.gif
    :alt: mnist GPU bar
    :width: 100 %

Below, we reuse the **HourLongWork** work defined in the previous example, but modify the **RootFlow**
to run two **HourLongWork** works in a parallel way.

.. literalinclude:: code_samples/quickstart/app/app_0.py
    :emphasize-lines: 21

Above, both ``child_work_1`` and ``child_work_2`` are long-running works that are executed
asynchronously in parallel.

When running the above app, we see the following logs:

.. code-block:: console

    $ lightning_app run app docs/source/code_samples/quickstart/app/app_0.py
      INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
      # After you have clicked `run` on the UI.
      0.0, 0.0
      ...
      0.0003, 0.0003
      ...
      1.0, 1.0
      ...
      1 hour later `child_work_1` started!
      1 hour later `child_work_2` started!
      0.0, 0.0
      ...
      0.0003, 0.0003
      ...
      1.0, 1.0
      1 hour later `child_work_1` started!
      1 hour later `child_work_2` started!
      ...

----

***********
Next Steps
***********

To keep learning about Lightning, build a :ref:`ui_and_frontends`.
