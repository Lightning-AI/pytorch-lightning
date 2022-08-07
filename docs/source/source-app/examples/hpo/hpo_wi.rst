:orphan:

##########################################
Step 2: Add the flow to your HPO component
##########################################

**Audience:** Users who want to understand how to implement HPO training from scratch with Lightning.

**Prereqs:** Level 17+

----

Thanks to the simplified version, you should have a good grasp on how to implement HPO with Optuna.

As the :class:`~lightning_app.core.app.LightningApp` handles the Infinite Loop,
it has been removed from within the run method of the HPORootFlow.

However, the ``run`` method code is the same as the one defined above.

.. literalinclude:: ../../../examples/app_hpo/app_wo_ui.py
    :language: python

The ``ObjectiveWork`` is sub-classing
the built-in :class:`~lightning_app.components.python.TracerPythonScript`
which enables launching scripts and more.

.. literalinclude:: ../../../examples/app_hpo/objective.py
    :language: python

Finally, let's add the ``HiPlotFlow`` component to visualize our hyperparameter optimization.

The metric and sampled parameters are added to the ``self.hi_plot.data`` list, enabling
updates to the dashboard in near-realtime.

.. literalinclude:: ../../../examples/app_hpo/app_wi_ui.py
    :diff: ../../../examples/app_hpo/app_wo_ui.py

Here is the associated code with the ``HiPlotFlow`` component.

In the ``render_fn`` method, the state of the ``HiPlotFlow`` is passed.
The ``state.data`` is accessed as it contains the metric and sampled parameters.

.. literalinclude:: ../../../examples/app_hpo/hyperplot.py

Run the HPO application with the following command:

.. code-block:: console

    $ lightning run app examples/app_hpo/app_wi_ui.py
      INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
      {0: ..., 1: ..., ..., 5: ...}

Here is what the UI looks like when launched:

.. image:: https://pl-flash-data.s3.amazonaws.com/assets_lightning/hpo_ui_2.gif
  :width: 100 %
  :alt: Alternative text
