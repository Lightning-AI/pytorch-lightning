:orphan:

.. _delete_cluster:


*******************************
Delete a Lightning BYOC cluster
*******************************

Once you no longer need a Lightning cluster you can delete it with the following command:

.. code:: bash

   lightning delete cluster <cluster-name>

Deleting a cluster will remove any apps data from Lighting (including logs and metadata) and all reources associated with the cluster. Any artifacts created in the object storage of your cluster will not be deleted.

.. warning:: Using the ``--force`` parameter when deleting a cluster does not clean up any resources managed by Lightning AI. Check your cloud provider to verify that existing cloud resources are deleted.

.. warning:: This process may take a few minutes to complete, but once started it CANNOT be rolled back. Deletion permanently removes not only the BYOC cluster from being managed by Lightning AI, but tears down every BYOC resource Lightning AI managed (for that cluster id) in the host cloud. All object stores, container registries, logs, compute nodes, volumes, etc. are deleted and cannot be recovered.

.. warning::

   Under the hood the deletion selects cloud provider resources via the tags
   `lightning/cluster` and
   `kubernetes.io/cluster/<name>`

   Do not use these tags in any cloud resources you create yourself, as they will be subject to deletion when the cluster is deleted.
