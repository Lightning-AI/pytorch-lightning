.. _multi-gpu-training:

Multi-GPU training
==================
Lightning supports multiple ways of doing distributed training.

Preparing your code
-------------------
To train on CPU/GPU/TPU without changing your code, we need to build a few good habits :)

Delete .cuda() or .to() calls
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Delete any calls to .cuda() or .to(device).

.. code-block:: python

    # before lightning
    def forward(self, x):
        x = x.cuda(0)
        layer_1.cuda(0)
        x_hat = layer_1(x)

    # after lightning
    def forward(self, x):
        x_hat = layer_1(x)

Init using type_as
^^^^^^^^^^^^^^^^^^
When you need to create a new tensor, use `type_as`.
This will make your code scale to any arbitrary number of GPUs or TPUs with Lightning

.. code-block:: python

    # before lightning
    def forward(self, x):
        z = torch.Tensor(2, 3)
        z = z.cuda(0)

    # with lightning
    def forward(self, x):
        z = torch.Tensor(2, 3)
        z = z.type_as(x)

Remove samplers
^^^^^^^^^^^^^^^
For multi-node or TPU training, in PyTorch we must use `torch.nn.DistributedSampler`. The
sampler makes sure each GPU sees the appropriate part of your data.

.. code-block:: python

    # without lightning
    def train_dataloader(self):
        dataset = MNIST(...)
        sampler = None

        if self.on_tpu:
            sampler = DistributedSampler(dataset)

        return DataLoader(dataset, sampler=sampler)

With Lightning, you don't need to do this because it takes care of adding the correct samplers
when needed.

.. code-block:: python

    # with lightning
    def train_dataloader(self):
        dataset = MNIST(...)
        return DataLoader(dataset)

Distributed modes
-----------------
Lightning allows multiple ways of training

- Data Parallel (`distributed_backend='dp'`) (multiple-gpus, 1 machine)
- DistributedDataParallel (`distributed_backend='ddp'`) (multiple-gpus across many machines).
- DistributedDataParallel2 (`distributed_backend='ddp2'`) (dp in a machine, ddp across machines).
- TPUs (`num_tpu_cores=8|x`) (tpu or TPU pod)

Data Parallel (dp)
^^^^^^^^^^^^^^^^^^
`DataParallel <https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel>`_ splits a batch across k GPUs. That is, if you have a batch of 32 and use dp with 2 gpus,
each GPU will process 16 samples, after which the root node will aggregate the results.

.. code-block:: python

    # train on 1 GPU (using dp mode)
    trainer = pl.Trainer(gpus=2, distributed_backend='dp')

Distributed Data Parallel
^^^^^^^^^^^^^^^^^^^^^^^^^
`DistributedDataParallel <https://pytorch.org/docs/stable/nn.html#distributeddataparallel>`_ works as follows.

1. Each GPU across every node gets its own process.

2. Each GPU gets visibility into a subset of the overall dataset. It will only ever see that subset.

3. Each process inits the model.

.. note:: Make sure  to set the random seed so that each model inits  with the same weights

4. Each process performs a full forward and backward pass in parallel.

5. The gradients are synced and averaged across all processes.

6. Each process updates its optimizer.

.. code-block:: python

    # train on 8 GPUs (same machine (ie: node))
    trainer = pl.Trainer(gpus=8, distributed_backend='ddp')

    # train on 32 GPUs (4 nodes)
    trainer = pl.Trainer(gpus=8, distributed_backend='ddp', num_nodes=4)

Distributed Data Parallel 2
^^^^^^^^^^^^^^^^^^^^^^^^^^^
In certain cases, it's advantageous to use all batches on the same machine instead of a subset.
For instance you might want to compute a NCE loss where it pays  to have more negative samples.

In  this case, we can use ddp2 which behaves like dp in a machine and ddp across nodes. DDP2 does the following:

1. Copies a subset of the  data to each node.

2. Inits a model on each node.

3. Runs a forward and backward pass using DP.

4. Syncs gradients across nodes.

5. Applies the optimizer updates.

.. code-block:: python

    # train on 32 GPUs (4 nodes)
    trainer = pl.Trainer(gpus=8, distributed_backend='ddp2', num_nodes=4)

DP/DDP2 caveats
^^^^^^^^^^^^^^^
In DP and DDP2 each GPU within a machine sees a portion of a batch.
DP and ddp2 roughly do the following:

.. code-block:: python

    def distributed_forward(batch, model):
        batch = torch.Tensor(32, 8)
        gpu_0_batch = batch[:8]
        gpu_1_batch = batch[8:16]
        gpu_2_batch = batch[16:24]
        gpu_3_batch = batch[24:]

        y_0 = model_copy_gpu_0(gpu_0_batch)
        y_1 = model_copy_gpu_0(gpu_1_batch)
        y_2 = model_copy_gpu_0(gpu_2_batch)
        y_3 = model_copy_gpu_0(gpu_3_batch)

        return [y_0, y_1, y_2, y_3]

So, when Lightning calls any of the `training_step`, `validation_step`, `test_step`
you will only be operating on one of those pieces.

.. code-block:: python

    # the batch here is a portion of the FULL batch
    def training_step(self, batch, batch_idx):
        y_0 = batch

For most metrics, this doesn't really matter. However, if you want
to add something to your computational graph (like softmax)
using all batch parts you can use the `training_step_end` step.

.. code-block:: python

    def training_step_end(self, outputs):
        # only use when  on dp
        outputs = torch.cat(outputs, dim=1)
        softmax = softmax(outputs, dim=1)
        out = softmax.mean()
        return out

In pseudocode, the full sequence is:

.. code-block:: python

    # get data
    batch = next(dataloader)

    # copy model and data to each gpu
    batch_splits = split_batch(batch, num_gpus)
    models = copy_model_to_gpus(model)

    # in parallel, operate on each batch chunk
    all_results = []
    for gpu_num in gpus:
        batch_split = batch_splits[gpu_num]
        gpu_model = models[gpu_num]
        out = gpu_model(batch_split)
        all_results.append(out)

    # use the full batch for something like softmax
    full out = model.training_step_end(all_results)

to illustrate why this is needed, let's look at dataparallel

.. code-block:: python

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(batch)

        # on dp or ddp2 if we did softmax now it would be wrong
        # because batch is actually a piece of the full batch
        return y_hat

    def training_step_end(self, batch_parts_outputs):
        # batch_parts_outputs has outputs of each part of the batch

        # do softmax here
        outputs = torch.cat(outputs, dim=1)
        softmax = softmax(outputs, dim=1)
        out = softmax.mean()

        return out

If `training_step_end` is defined it will be called regardless of tpu, dp, ddp, etc... which means
it will behave the same no matter the backend.

Validation and test step also have the same option when using dp

.. code-block:: python

        def validation_step_end(self, batch_parts_outputs):
            ...

        def test_step_end(self, batch_parts_outputs):
            ...

Implement Your Own Distributed (DDP) training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you need your own way to init PyTorch DDP you can override :meth:`pytorch_lightning.core.LightningModule.`.

If you also need to use your own DDP implementation, override:  :meth:`pytorch_lightning.core.LightningModule.configure_ddp`.
