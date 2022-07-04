############################
Level 4: Modify existing app
############################
**Audience:** Users who have already run a Lightning App locally or remote and want to modify it.

**Prereqs:** You've ran a Lightning App locally and the cloud.

----

***************
Change the code
***************
A Lightning App is simply organized Python code. To modify an existing Lightning App, simply change the code!
There's nothing more you need to do.

----

***************
Add a component
***************
A major superpower you get with Lightning Apps is modular workflows. This allows you to use our gallery
of opensource components to power your work. Find a component in the `component gallery <https://lightning.ai/components>`_ and add it to your
Lightning App.

If you need inspiration, here are some components you might want to use to extend your Lightning App:


.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Slack Messenger
   :description: Send a Slack notification when anything happens in your Lightning App.
   :button_link: https://lightning.ai/components/LJFITYeNBZ-Slack%20Messenger
   :col_css: col-md-4
   :height: 200
   :tag: monitoring

.. displayitem::
   :header: Lightning Serve
   :description: Make your model serving infrastructure resilient with over 7+ strategies such as canary, recreate, ramped and more.
   :button_link: https://lightning.ai/components/BA2slXIke9-Lightning%20Serve
   :col_css: col-md-4
   :height: 200
   :tag: production

.. displayitem::
   :header: Lightning HPO
   :description: Add a hyperparameter sweep to your Lightning App.
   :button_link: https://lightning.ai/components/BA2slXI093-Lightning%20HPO
   :col_css: col-md-4
   :height: 200
   :tag: research

.. displayitem::
   :header: Jupyter Notebook
   :description: Add a Jupyter Notebook to your Lightning App.
   :button_link: https://lightning.ai/components/cRH1UHnvBx-Jupyter%20Notebook
   :col_css: col-md-4
   :height: 200
   :tag: data science

.. displayitem::
   :header: Google BigQuery
   :description: Connect a big BigQuery dataset to your Lightning App.
   :button_link: https://lightning.ai/components/Mtt4fnRlUE-Google%20BigQuery
   :col_css: col-md-4
   :height: 200
   :tag: production

.. raw:: html

        </div>
    </div>

----

*********************
Install the component
*********************
To add a component, install the component first.

We'll use the Slack messaging component as example:

.. code:: bash

    lightning install component lightning/lit-slack-messenger


Now that the component is installed, make sure you add it to your requirements.txt

.. code:: bash

    echo 'git+https://github.com/Lightning-AI/LAI-slack-messenger.git@4aa91554f51baf56fc14316365c67fcc67b61e7d' > requirements.txt

----

*****************
Use the component
*****************
To use the component, simply import it and attach it to your Lightning App.

.. code:: python

    import lightning as L
    from lit_slack import SlackMessenger


    class YourComponent(L.LightningFlow):
        def __init__(self):
            super().__init__()
            self.slack_messenger = SlackMessenger(token="a-long-token", channel_id="A03CB4A6AK7")

        def run(self):
            self.slack_messenger.send_message("hello from ⚡ lit slack ⚡")


    app = L.LightningApp(YourComponent())
