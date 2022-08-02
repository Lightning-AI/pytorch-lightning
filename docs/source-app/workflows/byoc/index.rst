
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

Setup begins with configuring a cloud provider (today AWS, but more are coming soon) with credentials for
delegated access and an identity provider for secure access to the data plane.

Next, as part of the environment creation process, users configure networking,
security, and select Lightning cluster configuration options based on their use cases.

After submitting the request,the Lightning control plane creates the required cloud infrastructure
and sets up the environment with the Lightning cluster along with a Lightning Kubernetes operator.


*******************************
Create a Lightning BYOC cluster
*******************************

You must have your cloud configured before you try and create a BYOC cluster.

Create a Lightning BYOC cluster using the following command:

PLEASE PROVIDE AN EXAMPLE WITH CLOUD PROVIDER PARAMETERS

.. code:: python

   lightning create cluster <cluster-name> <cloud-provider-parameters>

Cluster names can only contain lowercase letters, numbers, and periodic hyphens ( - ).

.. note:: AWS is supported today, but support for other cloud providers is coming soon.

Args:

* cost_savings: Specifies if the cluster uses cost savings mode.

.. note:: In cost saving mode the number of compute nodes is reduced to one, reducing the cost for clusters with low utilization.

* cluster_name: The name of the cluster to be created

* role_arn: AWS IAM Role ARN used to provision resources

* region: AWS region containing compute resources

* external_id: AWS IAM Role external ID

* instance_types: AWS instance types supported by the cluster

* edit_before_creation: Enables interactive editing of requests before submitting it to Lightning AI.

* wait: Waits for the cluster to be in a RUNNING state. Only use this for debugging.

----

*******************************************
View a list of your Lightning BYOC clusters
*******************************************

.. code:: python

   lightning list clusters

*******************************
Delete a Lightning BYOC cluster
*******************************

Deletes a Lightning BYOC cluster. Lightning AI removes cluster artifacts and any resources running on the cluster.

.. warning:: Deleting a cluster does not clean up any resources managed by Lightning AI. Check your cloud provider to verify that existing cloud resources are deleted.

Deleting a run also deletes all Runs and Experiments that were started on the cluster.
Deletion permanently removes not only the record of all runs on a cluster, but all associated experiments, artifacts, metrics, logs, etc.

.. warning:: This process may take a few minutes to complete, but once started it CANNOT be rolled back. Deletion permanently removes not only the BYOC cluster from being managed by Lightning AI, but tears down every BYOC resource Lightning AI managed (for that cluster id) in the host cloud. All object stores, container registries, logs, compute nodes, volumes, etc. are deleted and cannot be recovered.

.. code:: python

   lightning delete cluster <cluster-name>
