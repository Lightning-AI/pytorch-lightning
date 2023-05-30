:orphan:

.. _create_cluster:


##################
Create AWS cluster
##################

**Audience:** Users looking to create a cluster to run Lightning Apps on their own private cloud infrastructure.

**Prereqs:** basic familiarity with cloud provider infrastructure management.

.. note:: This feature is currently available for early access! To create your own cluster `contact us <mailto:product@lightning.ai?subject=I%20want%20to%20run%20on%20my%20private%20cloud!>`_.


----

*******************************************
Step 1- Create roles and permissions on AWS
*******************************************

In this step you’ll be creating a role on your cloud provider that allows Lightning to manage resources on your behalf (for example, creating EC2 instances for your cluster).
To do this you can use the AWS CLI or the AWS management console.

You will only have to preform this step once, and the same role can be used to create multiple clusters.

----

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Create role with AWS CLI
   :description: Create role with AWS CLI
   :col_css: col-md-4
   :button_link: aws_cli.html
   :height: 180
   :tag: Basic

.. displayitem::
   :header: Create role with AWS console
   :description: Create role with AWS console
   :col_css: col-md-4
   :button_link: aws_console.html
   :height: 180
   :tag: Basic

.. raw:: html

        </div>
    </div>


----


****************************
Step 2- Get ARN for new role
****************************

To start a cluster, Lightning will need the external-id you set in step 1, and the ARN role. Click on your new role to find them (note: you need the ARN listed at the top of the page, not the one in “trusted entitles”).


Record the ARN and the external ID. You’ll need them for your next step.

-----

**************************************
Step 3-Create a Lightning BYOC cluster
**************************************

Now that you have created a role and policy on AWS, you can start creating Lightning clusters.

Create a Lightning BYOC cluster using the following command:

.. code:: bash

   lightning create cluster <cluster-name> --role-arn <ARN> --external-id <EXTERNAL-ID>

Here's an example:

.. code:: bash

   lightning create cluster my-byoc-cluster --role-arn arn:aws:iam::1234567890:role/lai-byoc --external-id dummy

.. note:: Cluster creation is going to take an hour or more after you run this command.
.. note:: Only us-east-1, us-east-2, us-west-1 and us-west-2 are supported today.


Parameters
==========

+------------------------+----------------------------------------------------------------------------------------------------+
|Parameter               | Description                                                                                        |
+========================+====================================================================================================+
| cluster_id             | The name of the cluster to be created.                                                             |
|                        |                                                                                                    |
|                        | Cluster names can only contain lowercase letters, numbers, and periodic hyphens ( - ).             |
+------------------------+----------------------------------------------------------------------------------------------------+
| role-arn               | AWS IAM Role ARN used to provision resources                                                       |
+------------------------+----------------------------------------------------------------------------------------------------+
| external-id            | AWS IAM Role external ID                                                                           |
|                        |                                                                                                    |
|                        | To read more on what the AWS external ID is and why it's useful go                                 |
|                        | `here <https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for-user_externalid.html>`_|
+------------------------+----------------------------------------------------------------------------------------------------+

----

*******************************************
View a list of your Lightning BYOC clusters
*******************************************

.. code:: bash

   lightning list clusters

---

******************************
Next: Run apps on your cluster
******************************

Once your cluster is running, you can start running Lightning apps on your cluster.

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Run apps on your cluster
   :description: Learn how to start apps on your Lightning cluster
   :button_link: run_on_cluster.html
   :col_css: col-md-12
   :height: 170

.. raw:: html

        </div>
    </div>
