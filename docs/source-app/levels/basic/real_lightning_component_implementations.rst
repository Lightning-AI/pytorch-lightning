###############################################
Level 2: Explore real component implementations
###############################################
**Audience:** Users who want to deeply understand what is possible with Lightning components.

**Prereqs:** You must have finished :doc:`level 1 <../basic/build_a_lightning_component>`.

----

***********************
Real component examples
***********************
Use this guide to understand what is happening in each type of component.
These are a few prototypical components. Since each component organizes
Python, you can build virtually infinite components for any use-case
you can think of.

----

*******************************
Ex: PyTorch + Lightning Trainer
*******************************
This example shows how to train PyTorch with the Lightning trainer on your machine
or cloud GPUs without code changes.

.. lit_tabs::
   :descriptions: import Lightning; We're using a demo LightningModule; Move your training code here (usually your main.py); Pass your component to the multi-node executor (it works on CPU or single GPUs also); Select the number of machines (nodes). Here we choose 4.; Choose from over 15+ machine types. This one has 4 v100 GPUs.; Initialize the App object that executes the component logic.
   :code_files: /levels/basic/hello_components/pl_multinode.py; /levels/basic/hello_components/pl_multinode.py; /levels/basic/hello_components/pl_multinode.py; /levels/basic/hello_components/pl_multinode.py;  /levels/basic/hello_components/pl_multinode.py; /levels/basic/hello_components/pl_multinode.py; /levels/basic/hello_components/pl_multinode.py;
   :highlights: 2; 4; 9-11; 14-17; 16; 17; 19
   :enable_run: true
   :tab_rows: 5
   :height: 420px

----

*********************************
Ex: Deploy a PyTorch API endpoint
*********************************
This example shows how to deploy PyTorch and create an API

.. lit_tabs::
   :descriptions: Shortcut to list dependencies without a requirements.txt file.; Import one of our serving components (high-performance ones are available on the enterprise tiers); Define the setup function to load your favorite pretrained models and do any kind of pre-processing.; Define the predict function which is called when the endpoint is hit.; Initialize the server and define the type of cloud machine to use.
   :code_files: /levels/basic/hello_components/deploy_model.py; /levels/basic/hello_components/deploy_model.py; /levels/basic/hello_components/deploy_model.py; /levels/basic/hello_components/deploy_model.py; /levels/basic/hello_components/deploy_model.py;
   :highlights: 1; 3; 10-12; 15-25; 28-30
   :enable_run: true
   :tab_rows: 4
   :height: 620px

----

*************************
Next: Save on cloud costs
*************************
Let's review key lightning features to help you run components more efficiently on the cloud
so you can save on cloud costs.

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Level 3: Save money on cloud costs
   :description: Explore key Lightning features that save you cloud costs and improve performance.
   :button_link: save_money_on_cloud_costs.html
   :col_css: col-md-12
   :height: 150
   :tag: 10 minutes

.. raw:: html

        </div>
    </div>
