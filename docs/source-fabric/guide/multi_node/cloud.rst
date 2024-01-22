:orphan:

##########################
Run in the Lightning Cloud
##########################

**Audience**: Users who don't want to waste time on cluster configuration and maintenance.


The Lightning AI cloud is a platform where you can build, train, finetune and deploy models without worrying about infrastructure, cost management, scaling, and other technical headaches.
In this guide, and within just 10 minutes, you will learn how to run a Fabric training script across multiple nodes in the cloud.


----


*************
Initial Setup
*************

First, create a free `Lightning AI account <https://lightning.ai/>`_.
You get free credits every month you can spend on GPU compute.
To use muliple machines, you need to be on the `Pro or Teams plan <https://lightning.ai/pricing>`_.


----


***************************************
Launch multi-node training in the cloud
***************************************

**Step 1:** Start a new studio.

.. video:: https://pl-public-data.s3.amazonaws.com/assets_lightning/fabric/videos/start-studio-for-mmt.mp4
    :width: 800
    :loop:
    :muted:

|

**Step 2:** Bring your code into the studio. You can clone a GitHub repo, drag and drop local files, or use the following demo example:

.. collapse:: Code Example

    .. code-block:: python

        import lightning as L
        import torch
        import torch.nn.functional as F
        from lightning.pytorch.demos import Transformer, WikiText2
        from torch.utils.data import DataLoader


        def main():
            L.seed_everything(42)

            fabric = L.Fabric()
            fabric.launch()

            # Data
            with fabric.rank_zero_first():
                dataset = WikiText2()

            train_dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

            # Model
            model = Transformer(vocab_size=dataset.vocab_size)

            # Optimizer
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

            model, optimizer = fabric.setup(model, optimizer)
            train_dataloader = fabric.setup_dataloaders(train_dataloader)

            for batch_idx, batch in enumerate(train_dataloader):
                input, target = batch
                output = model(input, target)
                loss = F.nll_loss(output, target.view(-1))
                fabric.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if batch_idx % 10 == 0:
                    fabric.print(f"iteration: {batch_idx} - loss {loss.item():.4f}")


        if __name__ == "__main__":
            main()

|

**Step 3:** Remove hardcoded accelerator settings if any and let Lightning automatically set them for you. No other changes are required in your script.

.. code-block:: python

    # These are the defaults
    fabric = L.Fabric(accelerator="auto", devices="auto")

    # DON'T hardcode these, leave them default/auto
    # fabric = L.Fabric(accelerator="cpu", devices=3)

|

**Step 4:** Install dependencies and download all necessary data. Test that your script runs in the studio first. If it runs in the studio, it will run in multi-node!

|

**Step 5:** Open the Multi-Machine Training (MMT) app. Type the command to run your script, select the machine type and how many machines you want to launch on. Click "Run" to start the job.

.. video:: https://pl-public-data.s3.amazonaws.com/assets_lightning/fabric/videos/lightning-ai-mmt-demo-fabric.mp4
    :width: 800
    :loop:
    :muted:


----


****************************
Bring your own cloud account
****************************

On the `Teams or Enterprise <https://lightning.ai/pricing>`_ tier, you can connect your own AWS account.



----


**********
Next steps
**********

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: Lightning Platform
    :description: Develop, Train and Deploy models on the cloud
    :col_css: col-md-4
    :button_link: https://lightning.ai
    :height: 150

.. raw:: html

        </div>
    </div>
