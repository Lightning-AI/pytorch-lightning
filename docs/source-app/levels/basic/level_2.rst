###################################
Level 2: Run Lightning Apps locally
###################################
**Audience:** New users who want to run a Lightning App on their machines

**Prereqs:** You already have the Lightning App code on your local machine.

----

************
Get the code
************
If you followed the instructions for **Level 1: Clone and Run**, you already have the code for the Lightning App locally. Otherwise please go back to `Level 1 <https://lightning.ai/lightning-docs/levels/basic/level_1.html>`_.

----

***********
Run locally
***********
Run the Lightning App on your local machine by using the following command:

.. code:: bash

    lightning run app app.py

Now you'll see the Lightning App start up on your local machine.

.. note:: At this time, you can only run one Lightning App locally at a time. **Submit a PR to unblock that!**

----

*************************
Run on any remote machine
*************************
Remember you can always SSH into any of your cloud machines on your university or enterprise cluster and run
Lightning App from there. However, you will be responsible for the auto-scaling of those Lightning Apps.
