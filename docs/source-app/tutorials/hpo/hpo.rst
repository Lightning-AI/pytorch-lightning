.. hpo:

******************
Build a Sweeps App
******************

Introduction
============

Traditionally, developing ML Products requires to choose among a large space of
hyperparameters while creating and training the ML models. Hyperparameter Optimization
(HPO) aims at finding a well-performing hyperparameter configuration of a given machine
learning model on a dataset at hand, including the machine learning model,
its hyperparameters and other data processing steps.

Thus, HPO frees the human expert from a tedious and error-prone hyperparameter tuning process.

As an example, in the famous `scikit-learn <https://scikit-learn.org/stable/>`_, hyperparameter are passed as arguments to the constructor of
the estimator classes such as ``C`` kernel for
`Support Vector Classifier <https://scikit-learn.org/stable/modules/classes.html?highlight=svm#module-sklearn.svm>`_, etc.

It is possible and recommended to search the hyper-parameter space for the best validation score.

An HPO search consists of:

* an objective method
* a parameter space defined
* a method for searching or sampling candidates

A naive method for sampling candidates is Grid Search which exhaustively considers all parameter combinations.

Hopefully, the field of Hyperparameter Optimization is very active and many methods have been developed to
optimize the time required to get strong candidates.

In the following tutorial, you will learn how to use Lightning together with the `Optuna <https://optuna.org/>`_.

`Optuna <https://optuna.org/>`_ is an open source hyperparameter optimization framework to automate hyperparameter search.
Out-of-the-box, it provides efficient algorithms to search large spaces and prune unpromising trials for faster results.

First, you will learn about the best practices on how to implement HPO without the Lightning Framework.
Secondly, we will dive into a working HPO application with Lightning and finally create a neat
`HiPlot UI <https://facebookresearch.github.io/hiplot/_static/demo/demo_basic_usage.html?hip.filters=%5B%5D&hip.color_by=%22dropout%22&hip.PARALLEL_PLOT.order=%5B%22uid%22%2C%22dropout%22%2C%22lr%22%2C%22loss%22%2C%22optimizer%22%5D>`_
for our application.

HPO Example Without Lightning
=============================

In the example below, we are emulating the Lightning Infinite Loop.

We are assuming have already defined an ``ObjectiveWork`` component which is responsible to run the objective method and track the metric through its state.

We are running ``TOTAL_TRIALS`` trials by series of  ``SIMULTANEOUS_TRIALS`` trials.
When starting, ``TOTAL_TRIALS`` ``ObjectiveWork`` are created.

The entire code runs within an infinite loop as it would within Lightning.

When iterating through the works, if the current ``objective_work`` hasn't started,
some new parameters are sampled from the Optuna Study with our custom distributions
and then passed to run method of the ``objective_work``.

The condition ``not objective_work.has_started`` will be ``False`` once ``objective_work.run()`` starts.

Also, the second condition will be ``True`` when the metric finally is defined within the state of the work,
which happens after many iterations of the Lightning Infinite Loop.

Finally, once the current ``SIMULTANEOUS_TRIALS`` have both registered their
metric to the Optuna Study, simply increments ``NUM_TRIALS`` by ``SIMULTANEOUS_TRIALS`` to launch the next trials.

.. literalinclude:: ./hpo.py
    :language: python
    :emphasize-lines: 14, 16, 30-32, 37-40, 43, 46-47

Below, you can find the simplified version of the ``ObjectiveWork`` where the metric is randomly sampled using numpy.

In realistic use case, the work executes some user defined code.

.. literalinclude:: ./objective.py
    :language: python
    :emphasize-lines: 8, 19-21

Here are the logs produced when running the application above:

.. code-block:: console

    $ python docs/source/tutorials/hpo/hpo.py
      INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
      # After you have clicked `run` on the UI.
      [I 2022-03-01 12:32:50,050] A new study created in memory with name: ...
      {0: 13.994859806481264, 1: 59.866743330127825, ..., 5: 94.65919769609225}

In the cloud, here is an animation how this application works:

.. image:: https://pl-flash-data.s3.amazonaws.com/assets_lightning/hpo.gif
    :alt: Animation showing how to HPO works UI in a distributed manner.


HPO Example With Lightning
==========================

Thanks the simplified version, you should have a good grasp on how to implement HPO with Optuna

As the :class:`~lightning_app.core.app.LightningApp` handles the Infinite Loop,
it has been removed from within the run method of the HPORootFlow.

However, the run method code is the same as the one defined above.

.. literalinclude:: ../../../../examples/hpo/app_wo_ui.py
    :language: python
    :emphasize-lines: 5, 17-23, 52-59

The ``ObjectiveWork`` is sub-classing
the built-in :class:`~lightning_app.components.python.TracerPythonScript`
which enables to launch scripts and more.

.. literalinclude:: ../../../../examples/hpo/objective.py
    :language: python
    :emphasize-lines: 9, 11, 15, 21-30, 40-46

Finally, let's add ``HiPlotFlow`` component to visualize our hyperparameter optimization.

The metric and sampled parameters are added to the ``self.hi_plot.data`` list, enabling
to get the dashboard updated in near-realtime.

.. literalinclude:: ../../../../examples/hpo/app_wi_ui.py
    :diff: ../../../../examples/hpo/app_wo_ui.py

Here is the associated code with the ``HiPlotFlow`` component.

In the ``render_fn`` method, the state of the ``HiPlotFlow`` is passed.
The ``state.data`` is accessed as it contains the metric and sampled parameters.

.. literalinclude:: ../../../../examples/hpo/hyperplot.py

Run the HPO application with the following command:

.. code-block:: console

    $ lightning run app examples/hpo/app_wi_ui.py
      INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
      {0: ..., 1: ..., ..., 5: ...}

Here is how the UI looks like when launched:

.. image:: https://pl-flash-data.s3.amazonaws.com/assets_lightning/hpo_ui_2.gif
  :width: 100 %
  :alt: Alternative text
