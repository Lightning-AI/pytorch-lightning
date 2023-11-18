###########################################
Level 7: Share variables between components
###########################################
**Audience:** Users who want to share variables and files across Lightning components.

**Prereqs:** You must have finished `intermediate level 5+ <run_lightning_work_in_parallel.rst>`_.

----

****************************************
Send a variable from Flow to a Component
****************************************
When a variable is defined on the LightningFlow (orchestrator), and
then it's passed into functions for the work components, under the hood
Lightning sends the variables across the machines for you automatically.

.. lit_tabs::
   :descriptions: Remember this component may live on its own machine; The flow may be on a separate machine as well; This variable is on the flow machine; When passed to the work component, it is actually sent across the network under the hood.; When it prints here, it prints on the work component machine (not the flow machine); The second string was directly created on machine 1
   :code_files: ./scripts/comms_1.py; ./scripts/comms_1.py; ./scripts/comms_1.py; ./scripts/comms_1.py; ./scripts/comms_1.py; ./scripts/comms_1.py
   :highlights: 4-7; 9-16; 15; 16; 6; 7;
   :enable_run: true
   :tab_rows: 3
   :height: 380px

|

.. collapse:: CLI output

    .. code-block::

        $ lightning run app app.py --open-ui=false

        Your Lightning App is starting. This won't take long.
        INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
        MACHINE 1: this string came from machine 0: "hello from machine 0"
        MACHINE 1: this string is on machine 1

|

In this example, we learned that we can send variables to components like in regular Python.
On a local machine, it will behave like Python. When the workflow is distributed on the cloud,
it makes network calls under the hood, but still functions like Python to you.

----

**************************************
Send a variable between two components
**************************************
A majority of workflows (especially in ML), require components to respond to a change in a component
likely running on a separate machine or even cluster.

Example Continuous deployment: Every time a model saves a checkpoint, we redeploy a model:

.. lit_tabs::
   :descriptions: Define a component that simulates training; Define a component that simulates deployment; Training will happen in parallel over a long period; The deployment server also runs in parallel forever; Start training in parallel (could take months); Whenever the model has a checkpoint deploy; When the checkpoint is updated, model re-deploys
   :code_files: ./scripts/two_work_comms.py; ./scripts/two_work_comms.py; ./scripts/two_work_comms.py; ./scripts/two_work_comms.py; ./scripts/two_work_comms.py; ./scripts/two_work_comms.py; ./scripts/two_work_comms.py
   :highlights: 5-18; 20-22; 27; 28; 31; 32, 33; 33
   :enable_run: true
   :tab_rows: 3
   :height: 690px

|

.. collapse:: CLI output:

    .. code::

        $ lightning run app app.py --open-ui=false

        Your Lightning App is starting. This won't take long.
        INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
        step=0: fake_loss=100000.0
        TRAIN COMPONENT: saved new checkpoint: /some/path/step=0_fake_loss=100000.0
        step=1: fake_loss=1.0
        DEPLOY COMPONENT: load new model from checkpoint: /some/path/step=0_fake_loss=100000.0
        step=2: fake_loss=0.5
        step=3: fake_loss=0.3333
        step=4: fake_loss=0.25
        step=5: fake_loss=0.2
        step=6: fake_loss=0.1667
        step=7: fake_loss=0.1429
        step=8: fake_loss=0.125
        step=9: fake_loss=0.1111
        step=10: fake_loss=0.1
        TRAIN COMPONENT: saved new checkpoint: /some/path/step=10_fake_loss=0.1
        DEPLOY COMPONENT: load new model from checkpoint: /some/path/step=10_fake_loss=0.1
        step=11: fake_loss=0.0909
        step=12: fake_loss=0.0833
        step=13: fake_loss=0.0769
        step=14: fake_loss=0.0714
        step=15: fake_loss=0.0667
        step=16: fake_loss=0.0625
        step=17: fake_loss=0.0588
        step=18: fake_loss=0.0556
        step=19: fake_loss=0.0526
        step=20: fake_loss=0.05
        TRAIN COMPONENT: saved new checkpoint: /some/path/step=20_fake_loss=0.05
        DEPLOY COMPONENT: load new model from checkpoint: /some/path/step=20_fake_loss=0.05
        step=21: fake_loss=0.0476

----

********************************************
Send a large variable between two components
********************************************
For large variables such as arrays, tensors, embeddings and so on, use Payload to enable
transferring them across components.

.. lit_tabs::
   :descriptions: Let's define a component to simulate generating embeddings (from a DB, feature store, etc...); This component simulates a server that will use the embeddings.; Run the component to generate the embeddings; Simulate embeddings as an array. Here you would query a DB, load from a feature store or disk or even use a neural network to extract the embedding.; Allow the embeddings to be transferred efficiently by wrapping them in the Payload object.; Pass the variable to the EmbeddingServer (just the pointer).; The data gets transferred once you use the .value attribute in the other component.
   :code_files: ./scripts/toy_payload.py; ./scripts/toy_payload.py; ./scripts/toy_payload.py; ./scripts/toy_payload.py; ./scripts/toy_payload.py; ./scripts/toy_payload.py; ./scripts/toy_payload.py;
   :highlights: 5-13; 15-19; 28; 12; 13; 29; 18
   :enable_run: true
   :tab_rows: 3
   :height: 600px

|

.. collapse:: CLI output

    .. code::

            $ lightning run app app.py --open-ui=false

            Your Lightning App is starting. This won't take long.
            INFO: Your app has started. View it in your browser: http://127.0.0.1:7501/view
            PROCESSOR: Generating embeddings...
            SERVER: Using embeddings from processor <lightning.app.storage.payload.Payload object at 0x123383d60>
            serving embeddings sent from EmbeddingProcessor:  [[1, 2, 3], [2, 3, 4]]

|

The payload object keeps the data on the machine and passes a pointer
to the data around the app until the data is needed by a component.

----

******************************************
Next steps: Share files between components
******************************************
Now that you know how to run components in parallel, we'll learn to share variables
across components to simplify complex workflows.

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Level 8: Share files between components
   :description: Learn to share files between components.
   :col_css: col-md-12
   :button_link: share_files_between_components.html
   :height: 150
   :tag: 10 minutes

.. raw:: html

        </div>
    </div>
