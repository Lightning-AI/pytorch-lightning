:orphan:

.. _aws_cli:


############################
Create AWS role with AWS CLI
############################

1. Install AWS CLI (see instructions `here <https://aws.amazon.com/cli/>`_).

2. Protect your role by creating a hard to guess password that will be used to authenticate Lightning (You will need to pass it to Lightning for authentication). In our example we will use `dummy`.

3. Create a role called `lightning-cloud` using the following command (replace <YOUR-HARD-TO-GUESS-PASSWORD> with your own):

.. code:: bash

   aws iam create-role \
     --role-name lightning-cloud \
     --assume-role-policy-document '{"Statement":[{"Action":"sts:AssumeRole","Effect": "Allow", "Principal": {"AWS": "arn:aws:iam::748115360335:root"}, "Condition": {"StringEquals": {"sts:ExternalId": "<YOUR-HARD-TO-GUESS-PASSWORD>"}}}]}' \
     --description " " \
     --max-session-duration 43200

4. Create a file `iam-policy.json` with the following permissions required for Lightning to manage cloud infrastructure for you:

.. code:: json

  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Action": [
          "autoscaling:*",
          "cloudwatch:*",
          "codebuild:*",
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

5. Create a IAM policy and associate it with the role we just created, and pass in the path to your new file:

.. code:: bash

   aws iam create-policy \
     --policy-name lightning-cloud \
     --description "policy granting lightning controlplane permissions" \
     --policy-document file:///my_dir/iam-policy.json

6. Fetch the role ARN so you can attach the policy:

.. code:: bash

   aws iam get-role --role-name lightning-cloud --output json --query Role.Arn

7. Attach the policy to the IAM role you just created:

.. code:: bash

   aws iam attach-role-policy \
     --role-name lightning-cloud \
     --policy-arn arn:aws:iam::1234567890:policy/lightning-cloud

------

**********************
Next: Create a cluster
**********************

You are now ready to create a Lightning cluster!

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Create cluster
   :description: Create an AWS cluster for running ligthning apps, skip to step 2
   :button_link: create_cluster.html
   :col_css: col-md-12
   :height: 170

.. raw:: html

        </div>
    </div>
