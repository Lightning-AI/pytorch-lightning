
#################################
Run Apps on your own cloud (BYOC)
#################################

**Audience:** Users looking to run Lightning Apps on their own private cloud infrastructure. The following document assumes
basic familiarity with cloud provider infrastructure.

----

*******************
A bit of background
*******************

BYOC - Bring Your Own Cloud, is an alternate deployment model to Lightning Cloud (fully managed SaaS).
BYOC separates the control- and dataplane. The dataplane, that includes Lightning clusters, services and Lightning Apps,
reside inside the user’s own cloud provider account. The controlplane resides on Lightning Cloud.

Using BYOC requires configuring of a cloud provider to grant Lightning Cloud permissions to manage
infrastructure on your behalf. Once the access has been configured, Lightning Cloud controlplane will take over,
managing the lifecycle of the cloud infrastructure required to run Lightning Apps.

Only AWS is supported as of now.

**************************
Configuring access for AWS
**************************

To grant the Lightning controlplane access to your AWS account you need to configure an IAM role and establish a
cross account trust relationship. The lightning controlplane runs on AWS account `748115360335`, so the trust relationship
will look like this:

.. code:: json

   {
       "Statement": [
           {
               "Action": "sts:AssumeRole",
               "Effect": "Allow",
               "Principal": {
                   "AWS": "arn:aws:iam::748115360335:root"
               },
               "Condition": {
                   "StringEquals": {
                       "sts:ExternalId": "dummy"
                   }
               }
           }
       ]
   }

To follow AWS security practice, we're also using an external id.
The external id should be set to a random value and only be used for lightning cloud. Above example uses 'dummy'.

Now we'll create an IAM role called  `lightning-cloud` using above trust relationship:

.. code:: bash

   aws iam create-role \
     --role-name lightning-cloud \
     --assume-role-policy-document '{"Statement":[{"Action":"sts:AssumeRole","Effect": "Allow", "Principal": {"AWS": "arn:aws:iam::748115360335:root"}, "Condition": {"StringEquals": {"sts:ExternalId": "dummy"}}}]}' \
     --description "grant lightning controlplane access" \
     --max-session-duration 43200

Lightning controlplane will assume this IAM role to manage your cloud infrastructure.
Next, you need to grant permissions to the IAM role to enable the Lightning controlplane to manage cloud infrastructure
for you:

.. code:: json

  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Action": [
          "autoscaling:*",
          "cloudwatch:*",
          "ec2:*",
          "ecr:*",
          "eks:*",
          "elasticloadbalancing:*",
          "events:*",
          "guardduty:*",
          "iam:*",
          "logs:*",
          "route53resolver:*",
          "s3:*",
          "sns:*",
          "sqs:*",
          "tag:GetResources",
          "resource-groups:SearchResources"
        ],
        "Effect": "Allow",
        "Resource": "*"
      },
      {
        "Effect": "Allow",
        "Action": "iam:CreateServiceLinkedRole",
        "Resource": "*",
        "Condition": {
          "StringLike": {
            "iam:AWSServiceName": [
              "guardduty.amazonaws.com",
              "malware-protection.guardduty.amazonaws.com"
            ]
          }
        }
      },
      {
        "Effect": "Allow",
        "Action": "iam:CreateServiceLinkedRole",
        "Resource": "*",
        "Condition": {
          "StringEquals": {
            "iam:AWSServiceName": [
              "autoscaling.amazonaws.com",
              "ec2scheduled.amazonaws.com",
              "elasticloadbalancing.amazonaws.com",
              "spot.amazonaws.com",
              "spotfleet.amazonaws.com",
              "transitgateway.amazonaws.com"
            ]
          }
        }
      }
    ]
  }

Save this into a file, and create a IAM policy and associate it with the role we just created:

.. code:: bash

   aws iam create-policy \
     --policy-name lightning-cloud \
     --description "policy granting lightning controlplane permissions" \
     --policy-document file:///tmp/iam-policy.json

Lastly, attach the policy to the IAM role you just created:

.. code:: bash

   aws iam attach-role-policy \
     --role-name lightning-cloud \
     --policy-arn arn:aws:iam::158793097533:policy/lightning-cloud

You are now ready to create a BYOC cluster in your own AWS account!

Reach out to support@lightning.ai if you want to use terraform or CloudFormation to provision these resources.

*******************************
Create a Lightning BYOC cluster
*******************************

You must have your cloud configured before you try and create a BYOC cluster.

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

* cluster_id: The name of the cluster to be created

.. note:: Cluster names can only contain lowercase letters, numbers, and periodic hyphens ( - ).

----

Parameters
^^^^^^^^^^

+------------------------+----------------------------------------------------------------------------------------------------+
|Parameter               | Description                                                                                        |
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
| async                  | Cluster creation will happen in the background.                                                    |
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

Deletion retains any artifacts created in the object storage of your cluster.

.. warning:: This process may take a few minutes to complete, but once started it CANNOT be rolled back. Deletion permanently removes not only the BYOC cluster from being managed by Lightning AI, but tears down every BYOC resource Lightning AI managed (for that cluster id) in the host cloud. All object stores, container registries, logs, compute nodes, volumes, etc. are deleted and cannot be recovered.

.. code:: bash

   lightning delete cluster <cluster-name>

.. warning::

   Under the hood the deletion selects cloud provider resources via the tags
   `lightning/cluster` and
   `kubernetes.io/cluster/<name>`

   Do not use these tags in any cloud resources you create yourself, as they will be subject to deletion when the cluster is deleted.
