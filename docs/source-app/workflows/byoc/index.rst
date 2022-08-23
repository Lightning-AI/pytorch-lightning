
#################################
Run Apps on your own cloud (BYOC)
#################################

**Audience:** Users looking to run Lightning Apps on their own private cloud.

----

*******************
A bit of background
*******************

BYOC - Bring Your Own Cloud, is an alternate deployment model to Lightning Cloud (fully managed SaaS).
BYOC separates the control and data plane. The data plane, that includes
Lightning clusters, services and Lightning Apps, reside inside the user’s VPC.
The control plane resides on Lightning Cloud.

Setup begins with configuring a cloud provider (today AWS, but more are coming soon) with your personal credentials for
delegated access and an identity provider for secure access to the data plane.

Next, as part of the environment creation process, you can configure networking,
security, and select among cluster configuration options based on their own use cases.

After submitting a cluster creation request, the Lightning Control Plane creates the required cloud infrastructure on the user account. This
sets up a new Lightning Cluster along with a Lightning Kubernetes Operator.


*******************************
Create a Lightning BYOC cluster
*******************************

You must have your cloud configured before you try and create a BYOC cluster.

And to make your life a little easier, we've made a `Terraform module to help with that <https://github.com/Lightning-AI/terraform-aws-lightning-byoc>`_.

Create a Lightning BYOC cluster using the following command:

.. code:: bash

   lightning create cluster <cluster-name> <cloud-provider-parameters>

Here's an example:

.. code:: bash

   lightning create cluster my-byoc-cluster --provider aws --role-arn arn:aws:iam::1234567890:role/lai-byoc --external-id dummy --region us-west-2 --instance-types t3.xlarge --enable-performance

.. note:: Cluster creation is going to take an hour or more after you run this command.

----

Arguments
^^^^^^^^^

* cluster_name: The name of the cluster to be created

.. note:: Cluster names can only contain lowercase letters, numbers, and periodic hyphens ( - ).

----

Parameters
^^^^^^^^^^

+------------------------+----------------------------------------------------------------------------------------------------+
|Parameter               | Descritption                                                                                       |
+========================+====================================================================================================+
| provider               | The cloud provider where your cluster is located.                                                  |
|                        |                                                                                                    |
|                        | AWS is supported today, but support for other cloud providers is coming soon.                      |
+------------------------+----------------------------------------------------------------------------------------------------+
| role-arn               | AWS IAM Role ARN used to provision resources                                                       |
+------------------------+----------------------------------------------------------------------------------------------------+
| external-id            | AWS IAM Role external ID                                                                           |
|                        |                                                                                                    |
|                        | To read more on what the AWS external ID is and why it's useful go                                 |
|                        | `here <https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for-user_externalid.html>`_|
+------------------------+----------------------------------------------------------------------------------------------------+
| region                 | AWS region containing compute resources                                                            |
+------------------------+----------------------------------------------------------------------------------------------------+
| instance-types         | Instance types that you want to support, for computer jobs within the cluster.                     |
|                        |                                                                                                    |
|                        | For now, this is the AWS instance types supported by the cluster.                                  |
+------------------------+----------------------------------------------------------------------------------------------------+
| enable-performance     | Specifies if the cluster uses cost savings mode.                                                   |
|                        |                                                                                                    |
|                        | In cost saving mode the number of compute nodes is reduced to one, reducing the cost for clusters  |
|                        | with low utilization.                                                                              |
+------------------------+----------------------------------------------------------------------------------------------------+
| edit-before-creation   | Enables interactive editing of requests before submitting it to Lightning AI.                      |
+------------------------+----------------------------------------------------------------------------------------------------+
| wait                   | Waits for the cluster to be in a RUNNING state. Only use this for debugging.                       |
+------------------------+----------------------------------------------------------------------------------------------------+

----

*******************************************
View a list of your Lightning BYOC clusters
*******************************************

.. code:: bash

   lightning list clusters

----

*******************************
Delete a Lightning BYOC cluster
*******************************

Deletes a Lightning BYOC cluster. Lightning AI removes cluster artifacts and any resources running on the cluster.

.. warning:: Using the ``--force`` parameter when deleting a cluster does not clean up any resources managed by Lightning AI. Check your cloud provider to verify that existing cloud resources are deleted.

Deletion permanently removes not only the record of all runs on a cluster, but all associated artifacts, metrics, logs, etc.

.. warning:: This process may take a few minutes to complete, but once started it CANNOT be rolled back. Deletion permanently removes not only the BYOC cluster from being managed by Lightning AI, but tears down every BYOC resource Lightning AI managed (for that cluster id) in the host cloud. All object stores, container registries, logs, compute nodes, volumes, etc. are deleted and cannot be recovered.

.. code:: bash

   lightning delete cluster <cluster-name>
