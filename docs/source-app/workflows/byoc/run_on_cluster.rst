:orphan:

.. _run_on_cluster:


*********************************
Run app on your Lightning Cluster
*********************************

Once your cluster is running, you can run any Lightning App on your cluster. To run an App on the Lightning BYOC cluster, use ``--cloud –cluster-id <cluste-id>`` in the command:

.. code:: bash

   lightning run app app.py --cloud --cluster-id <cluster-id>

Here’s an example:

.. code:: bash

   lightning run app app.py --cloud --cluster-id my-byoc-cluster


View the status of your App using the following command:

.. code:: bash

   lightning list apps
