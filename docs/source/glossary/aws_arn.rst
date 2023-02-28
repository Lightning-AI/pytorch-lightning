:orphan:

.. _aws_arn:

#######
AWS ARN
#######

**Audience:** Users who want to run on their AWS account

**Level:** Intermediate

----

*******************
What is an AWS ARN?
*******************
An AWS Amazon Resource Name (ARN) are unique identifiers of Amazon resources (datasets, buckets, machines, clusters) with
customized access controls.

----

*************
Create an ARN
*************
To create an ARN, first install the AWS CLI

.. code:: bash

    # Linux
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install

    # MAC
    curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
    sudo installer -pkg AWSCLIV2.pkg -target /

    # WINDOWS
    msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi

Or `follow the AWS guide <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>`_.

Then enter the following commands:

.. code:: bash

    # TODO

----

***********************
Run on your AWS account
***********************
To run on your own AWS account, set up a Lightning cluster (here we name it pikachu):

.. code:: bash

   lightning create cluster pikachu --provider aws --role-arn arn:aws:iam::1234567890:role/lai-byoc --external-id dummy --region us-west-2

Run your code on the pikachu cluster by passing it into CloudCompute:

.. code:: python

   # app.py
   import lightning as L

   class LitWorker(L.LightningWork):
      def run(self):
         message = """
         ANY python code can run here such as:
         - train a model
         - launch a deployment server
         - label data
         - run a react app, dash app, streamlit app, etc...
         - start a jupyter notebook
         - subprocess.Popen('echo run any shell script, python scripts or non python files')
         """
         print(message)

   # uses 1 cloud GPU (or your own hardware)
   compute = L.CloudCompute('gpu', clusters=['pikachu'])
   app = L.LightningApp(LitWorker(cloud_compute=compute))
