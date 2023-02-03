.. _byoc:

#################################
Run Apps on your own cloud (BYOC)
#################################

**Audience:** Users looking to run Lightning Apps on their own private cloud infrastructure.

.. note:: This feature is currently available for early access! To create your own cluster `contact us <mailto:product@lightning.ai?subject=I%20want%20to%20run%20on%20my%20private%20cloud!>`_.


----

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
   :header: Create an AWS cluster
   :description: Create an AWS cluster
   :col_css: col-md-4
   :button_link: create_cluster.html
   :height: 180
   :tag: Basic

.. displayitem::
   :header: Run app on your cluster
   :description: How to run apps on your Lighnting Cluster
   :col_css: col-md-4
   :button_link: run_on_cluster.html
   :height: 180
   :tag: Basic

.. displayitem::
   :header: Delete a cluster
   :description: Delete a cluster
   :col_css: col-md-4
   :button_link: delete_cluster.html
   :height: 180
   :tag: Basic

.. raw:: html

        </div>
    </div>


----



********************
Why create a cluster
********************

You can use Lightning clusters to run Lightning apps on your own cloud provider account in order to protect your data and use your cloud provider's credits. The control for these clusters runs on the Lightning managed cloud, but the data plane, including the clusters, services, and apps, is located within your own cloud provider account.

Once the cluster is created, Lightning Cloud controlplane will take over,
managing the lifecycle of the cloud infrastructure required to run Lightning Apps.
