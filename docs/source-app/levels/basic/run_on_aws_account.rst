:orphan:

To run on your own AWS account, first `create an AWS ARN <../../glossary/aws_arn.rst>`_.

Next, set up a Lightning cluster (here we name it pikachu):

.. code:: bash

   # TODO: need to remove  --external-id dummy --region us-west-2
   lightning create cluster pikachu --provider aws --role-arn arn:aws:iam::1234567890:role/lai-byoc

Run your code on the pikachu cluster by passing it into CloudCompute:

.. code:: python

   compute = L.CloudCompute('gpu', clusters=['pikachu'])
   app = L.LightningApp(LitWorker(cloud_compute=compute))

.. warning::

   This feature is available only under early-access. Request access by emailing support@lightning.ai.
