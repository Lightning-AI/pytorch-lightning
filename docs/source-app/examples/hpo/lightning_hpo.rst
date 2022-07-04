:orphan:

################################
Re-use an existing HPO component
################################

**Audience:** Users who want to easily get started with HPO training.

**Prereqs:** Level 8+

----

*********************
Install Lightning HPO
*********************

Lightning HPO provides a Pythonic implementation for Scalable Hyperparameter Tuning
and relies on Optuna for providing state-of-the-art sampling hyper-parameters algorithms and efficient trial pruning strategies.

Find the `Lightning Sweeper App <https://lightning.ai/app/8FOWcOVsdf-Lightning%20Sweeper>`_ on `lightning.ai <https://lightning.ai/>`_ and its associated `Github repo <https://github.com/Lightning-AI/LAI-lightning-hpo-App>`_.

.. code-block:: bash

    lightning install app lightning/hpo

*********************
Lightning HPO Example
*********************

In this tutorial, we are going to convert `Optuna Efficient Optimization Algorithms <https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#sphx-glr-tutorial-10-key-features-003-efficient-optimization-algorithms-py>`_ into a Lightning App.

The Optuna example optimizes the value (example: learning-rate) of a ``SGDClassifier`` from ``sklearn`` trained over the `Iris Dataset <https://archive.ics.uci.edu/ml/datasets/iris>`_.

.. literalinclude:: ./optuna_reference.py
    :language: python


As you can see, several trials were pruned (stopped) before they finished all of the iterations.

.. code-block:: console

    A new study created in memory with name: no-name-4423c12c-22e1-4eaf-ba60-caf0020403c6
    Trial 0 finished with value: 0.07894736842105265 and parameters: {'alpha': 0.00020629773477269024}. Best is trial 0 with value: 0.07894736842105265.
    Trial 1 finished with value: 0.368421052631579 and parameters: {'alpha': 0.0005250149151047217}. Best is trial 0 with value: 0.07894736842105265.
    Trial 2 finished with value: 0.052631578947368474 and parameters: {'alpha': 5.9086862655635784e-05}. Best is trial 2 with value: 0.052631578947368474.
    Trial 3 finished with value: 0.3421052631578947 and parameters: {'alpha': 0.07177263583415294}. Best is trial 2 with value: 0.052631578947368474.
    Trial 4 finished with value: 0.23684210526315785 and parameters: {'alpha': 1.7451874636151302e-05}. Best is trial 2 with value: 0.052631578947368474.
    Trial 5 pruned.
    Trial 6 finished with value: 0.10526315789473684 and parameters: {'alpha': 1.4943994864178649e-05}. Best is trial 2 with value: 0.052631578947368474.
    Trial 7 pruned.
    Trial 8 pruned.
    Trial 9 pruned.
    Trial 10 pruned.
    Trial 11 pruned.
    Trial 12 pruned.
    Trial 13 pruned.
    Trial 14 pruned.
    Trial 15 pruned.
    Trial 16 finished with value: 0.07894736842105265 and parameters: {'alpha': 0.006166329613687364}. Best is trial 2 with value: 0.052631578947368474.
    Trial 17 pruned.
    Trial 18 pruned.
    Trial 19 pruned.

The example above has been re-organized in order to run as Lightning App.

.. literalinclude:: ./lightning_hpo_target.py
    :language: python

Now, your code can run at scale in the cloud, if needed, and it has a simple neat UI.

.. figure:: https://pl-flash-data.s3.amazonaws.com/assets_lightning/lightning_hpo_optimizer.png
    :alt: Lightning App UI
    :width: 100 %

As you can see, several trials were pruned (stopped) before they finished all of the iterations. Same as when using pure optuna.

.. code-block:: console

    A new study created in memory with name: no-name-a93d848e-a225-4df3-a9c3-5f86680e295d
    Trial 0 finished with value: 0.23684210526315785 and parameters: {'alpha': 0.006779437004523296}. Best is trial 0 with value: 0.23684210526315785.
    Trial 1 finished with value: 0.07894736842105265 and parameters: {'alpha': 0.008936151407006062}. Best is trial 1 with value: 0.07894736842105265.
    Trial 2 finished with value: 0.052631578947368474 and parameters: {'alpha': 0.0035836511240528008}. Best is trial 2 with value: 0.052631578947368474.
    Trial 3 finished with value: 0.052631578947368474 and parameters: {'alpha': 0.0005393218926409795}. Best is trial 2 with value: 0.052631578947368474.
    Trial 4 finished with value: 0.1578947368421053 and parameters: {'alpha': 6.572557493358585e-05}. Best is trial 2 with value: 0.052631578947368474.
    Trial 5 finished with value: 0.02631578947368418 and parameters: {'alpha': 0.0013953760106345603}. Best is trial 5 with value: 0.02631578947368418.
    Trail 6 pruned.
    Trail 7 pruned.
    Trail 8 pruned.
    Trail 9 pruned.
    Trial 10 finished with value: 0.07894736842105265 and parameters: {'alpha': 0.00555435554783454}. Best is trial 5 with value: 0.02631578947368418.
    Trail 11 pruned.
    Trial 12 finished with value: 0.052631578947368474 and parameters: {'alpha': 0.025624276147153992}. Best is trial 5 with value: 0.02631578947368418.
    Trial 13 finished with value: 0.07894736842105265 and parameters: {'alpha': 0.014613957457075546}. Best is trial 5 with value: 0.02631578947368418.
    Trail 14 pruned.
    Trail 15 pruned.
    Trail 16 pruned.
    Trial 17 finished with value: 0.052631578947368474 and parameters: {'alpha': 0.01028208215647372}. Best is trial 5 with value: 0.02631578947368418.
    Trail 18 pruned.
    Trail 19 pruned.
