:orphan:

.. _logging_basic:

#######################################
Track and Visualize Experiments (basic)
#######################################
**Audience:** Users who want to visualize and monitor their model development

----

*******************************
Why do I need to track metrics?
*******************************
In model development, we track values of interest such as the *validation_loss* to visualize the learning process for our models. Model development is like driving a car without windows, charts and logs provide the *windows* to know where to drive the car.

With Lightning, you can visualize virtually anything you can think of: numbers, text, images, audio. Your creativity and imagination are the only limiting factor.

----

*************
Track metrics
*************
Metric visualization is the most basic but powerful way of understanding how your model is doing throughout the model development process.

To track a metric, simply use the *self.log* method available inside the *LightningModule*

.. code-block:: python

    class LitModel(pl.LightningModule):
        def training_step(self, batch, batch_idx):
            value = self.global_step
            self.log("some_value", self.global_step)

To log multiple metrics at once, use *self.log_dict*

.. code-block:: python

    values = {"loss": loss, "acc": acc, "metric_n": metric_n}  # add more items if needed
    self.log_dict(values)

TODO: show plot of metric changing over time

----

View in the commandline
=======================

To view metrics in the commandline progress bar, set the *prog_bar* argument to True.

.. code-block:: python

    self.log(prog_bar=True)

TODO: need progress bar here

----

View in the browser
===================
To view metrics in the browser you need to use an *experiment manager* with these capabilities. By Default, Lightning uses Tensorboard which is free and opensource.

Tensorboard is already enabled by default

.. code-block:: python

    # every trainer already has tensorboard enabled by default
    trainer = Trainer()

To launch the tensorboard dashboard run the following command on the commandline.

.. code-block:: bash

    tensorboard --logdir=lightning_logs/

If you're using a notebook environment such as *colab* or *kaggle* or *jupyter*, launch Tensorboard with this command

.. code-block:: bash

    %reload_ext tensorboard
    %tensorboard --logdir=lightning_logs/

----

Accumulate a metric
===================
When *self.log* is called inside the *training_step*, it generates a timeseries showing how the metric behaves over time.

TODO: show chart

However, For the validation and test sets we are not generally interested in plotting the metric values per batch of data. Instead, we want to compute a summary statistic (such as average, min or max) across the full split of data.

When you call self.log inside the *validation_step* and *test_step*, Lightning automatically accumulates the metric and averages it once it's gone through the whole split (*epoch*).

.. code-block:: python

    def validation_step(self, batch, batch_idx):
        value = batch_idx + 1
        self.log("average_value", value)

TODO: show single point plotted

If you don't want to average you can also choose from ``{min,max,sum}`` by passing the *reduce_fx* argument.

.. code-block:: python

    # default function
    self.log(..., reduce_fx=torch.mean)

For other reductions, we recommend logging a :class:`torchmetrics.Metric` instance instead.

----

************
Track images
************
If your *experiment manager* supports image visualization, simply *log* the image with *self.log*

.. code-block:: python

    # (32 batch samples, 3 channels, 32 width, 32 height)
    image = torch.Tensor(32, 3, 28, 28)
    self.log("an_image", image)

----

**********
Track text
**********
If your *experiment manager* supports text visualization, simply *log* the text with *self.log*

.. code-block:: python

    text = "hello world"
    self.log("some_text", text)

# TODO: show screenshot

----

******************************
Configure the saving directory
******************************
By default, anything that is logged is saved to the current working directory. To use a different directory, set the *default_root_dir* argument in the Trainer.

.. code-block:: python

    Trainer(default_root_dir="/your/custom/path")
