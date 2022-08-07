:orphan:

###########################################################
Step 1: Implement an HPO component with the Lightning Works
###########################################################

**Audience:** Users who want to understand how to implement HPO training from scratch.

**Prereqs:** Level 17+

----

In the example below, we are emulating the Lightning Infinite Loop.

We are assuming we have already defined an ``ObjectiveWork`` component which is responsible to run the objective method and track the metric through its state.

.. literalinclude:: ./hpo.py
    :language: python

We are running ``TOTAL_TRIALS`` trials by series of  ``SIMULTANEOUS_TRIALS`` trials.
When starting, ``TOTAL_TRIALS`` ``ObjectiveWork`` are created.

The entire code runs within an infinite loop as it would within Lightning.

When iterating through the Works, if the current ``objective_work`` hasn't started,
some new parameters are sampled from the Optuna Study with our custom distributions
and then passed to run method of the ``objective_work``.

The condition ``not objective_work.has_started`` will be ``False`` once ``objective_work.run()`` starts.

Also, the second condition ``objective_work.has_told_study`` will be ``True`` when the metric
is defined within the state of the Work and has been shared with the study.

Finally, once the current ``SIMULTANEOUS_TRIALS`` have both registered their
metric to the Optuna Study, simply increment ``NUM_TRIALS`` by ``SIMULTANEOUS_TRIALS`` to launch the next trials.

Below, you can find the simplified version of the ``ObjectiveWork`` where the metric is randomly sampled using NumPy.

In a realistic use case, the Work executes some user-defined code.

.. literalinclude:: ./objective.py
    :language: python

Here are the logs produced when running the application above:

.. code-block:: console

    $ python docs/source-app/tutorials/hpo/hpo.py
      INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
      # After you have clicked `run` on the UI.
      [I 2022-03-01 12:32:50,050] A new study created in memory with name: ...
      {0: 13.994859806481264, 1: 59.866743330127825, ..., 5: 94.65919769609225}

The following animation shows how this application works in the cloud:

.. image:: https://pl-flash-data.s3.amazonaws.com/assets_lightning/hpo.gif
    :alt: Animation showing how to HPO works UI in a distributed manner.
