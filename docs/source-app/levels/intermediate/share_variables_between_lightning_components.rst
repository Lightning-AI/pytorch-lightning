###########################################
Level 6: Share variables between components
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
   :descriptions: Toy app with a single work; Remember this component may live on its own machine; The flow may be on a separate machine as well; This variable is on the flow machine; When passed to the work component, it is actually sent across the network under the hood.; When it prints here, it prints on the work component machine (not the flow machine); The second string was directly created on machine 1
   :code_files: ./scripts/comms_1.py; ./scripts/comms_1.py; ./scripts/comms_1.py; ./scripts/comms_1.py; ./scripts/comms_1.py; ./scripts/comms_1.py; ./scripts/comms_1.py
   :highlights: ; 4-7; 9-16; 15; 16; 6; 7;
   :app_id: abc123
   :tab_rows: 3
   :height: 430px

|

In this example, we learned that we can send variables to components like in regular Python.
On a local machine, it will behave like Python. When the workflow is distributed on the cloud,
it makes network calls under the hood, but still functions like Python to you.

.. collapse:: ML Example: TODO

    TODO

----

**************************************
Send a variable between two components
**************************************
A majority of workflows (especially in ML), require components to respond to a change in a component
likely running on a separate machine or even cluster.

In this example, every time a model saves a checkpoint, we redeploy a model:

.. lit_tabs::
   :descriptions: Define a component that simulates training; Define a component that simulates deployment; Training will happen in parallel over a long period; The deployment server also runs in parallel forever
   :code_files: ./scripts/two_work_comms.py; ./scripts/two_work_comms.py; ./scripts/two_work_comms.py; ./scripts/two_work_comms.py
   :highlights: 5-18; 20-22; 27; 28
   :app_id: abc123
   :tab_rows: 3
   :height: 690px

----

********************************************
Send a large variable between two components
********************************************
Payload.
