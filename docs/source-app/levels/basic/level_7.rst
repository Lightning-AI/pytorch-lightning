##################################
Level 7: Run on your cloud account
##################################
**Audience:** Users who want to run on their own cloud accounts

**Prereqs:** Users ran a Lightning App locally and/or the cloud.

----

****************************
What is the Lightning Cloud?
****************************
The Lightning Cloud is the platform that we've created to interface with the cloud providers. Today
the Lightning Cloud supports AWS.

.. note:: Support for GCP and Azure is coming soon!

To use the Lightning Cloud, you buy credits that are used to pay the cloud providers. If you want to run
on your own AWS credentials, please contact us (support@lightning.ai) so we can get your clusters set up for you.

----

****************************************
Run Lightning Apps on your cloud account
****************************************
To run Lightning Apps on your own account, you can simply SSH into the machines on your cloud account
and start the Lightning Apps there! This also works on any machine you have access to, such as your own
on-prem cluster, DGX machine, etc... However, you will have to manage your own auto-scaling
and distribition of work for complex Lightning Apps.

If you want to automate all that complexity, we allow you to create as many clusters as you
want, on the cloud provider of your choice, using Lightning Cloud. Once you've configured your clusters,
you can run your Lightning Apps on those clusters.

Free clusters on Lightning Cloud have limits of the number of machines you can run. To increase that limit,
please `contact us <support@lightning.ai>`_.
