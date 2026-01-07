Accumulated gradients run K small batches of size ``N`` before doing a backward pass. The effect is a large effective batch size of size ``KxN``, where ``N`` is the batch size.
Internally it doesn't stack up the batches and do a forward pass rather it accumulates the gradients for K batches and then do an ``optimizer.step`` to make sure the
effective batch size is increased but there is no memory overhead.

.. warning::

    When using distributed training for eg. DDP, with let's say with ``P`` devices, each device accumulates independently i.e. it stores the gradients
    after each ``loss.backward()`` and doesn't sync the gradients across the devices until we call ``optimizer.step()``. So for each accumulation
    step, the effective batch size on each device will remain ``N*K`` but right before the ``optimizer.step()``, the gradient sync will make the effective
    batch size as ``P*N*K``. For DP, since the batch is split across devices, the final effective batch size will be ``N*K``.

.. testcode::

    # DEFAULT (ie: no accumulated grads)
    trainer = Trainer(accumulate_grad_batches=1)

    # Accumulate gradients for 7 batches
    trainer = Trainer(accumulate_grad_batches=7)

Optionally, you can make the ``accumulate_grad_batches`` value change over time by using the :class:`~lightning.pytorch.callbacks.gradient_accumulation_scheduler.GradientAccumulationScheduler`.
Pass in a scheduling dictionary, where the key represents the epoch at which the value for gradient accumulation should be updated.

.. testcode::

        from lightning.pytorch.callbacks import GradientAccumulationScheduler

        # till 5th epoch, it will accumulate every 8 batches. From 5th epoch
        # till 9th epoch it will accumulate every 4 batches and after that no accumulation
        # will happen. Note that you need to use zero-indexed epoch keys here
        accumulator = GradientAccumulationScheduler(scheduling={0: 8, 4: 4, 8: 1})
        trainer = Trainer(callbacks=accumulator)

Note: Not all strategies and accelerators support variable gradient accumulation windows.

----

Gradient Accumulation with CrossEntropyLoss
=============================================

When using :class:`~torch.nn.CrossEntropyLoss` with gradient accumulation, ensure correct loss scaling by following these requirements:

1. **Batch Format**: Your dataloader must return a dictionary containing a ``"labels"`` key
2. **Loss Reduction**:  Compute loss with ``reduction="sum"`` instead of the default ``"mean"``
3. **Mask Token**:  Use ``-100`` for ignored/padded tokens (the PyTorch default ignored index)

Lightning will automatically normalize the accumulated loss based on the total number of valid targets across all micro-batches in the accumulation window.

.. testcode::  python

    class MyLightningModule(L.LightningModule):
        def training_step(self, batch, batch_idx):
            # batch is a dict with "labels" key from dataloader
            x = batch["input_ids"]
            x = x[:, :-1] # Example input processing
            labels = batch["labels"]
            logits = self.model(x)
            # Use reduction="sum" for correct accumulation
            loss = F.cross_entropy(logits, labels, reduction="sum")
            return loss

    # In your dataloader:
    def collate_fn(batch):
        # Return dict with "labels" key
        input_ids = torch.stack([item["input"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        return {"input_ids": input_ids, "labels": labels}

    dataloader = DataLoader(dataset, collate_fn=collate_fn)
    trainer = L.Trainer(accumulate_grad_batches=4)

..  warning::

    If your batch does not contain a "labels" key or does not follow the expected format, Lightning will fall back to the standard per-micro-batch averaging, which may result in different training dynamics compared to training with a single large batch.  This discrepancy is more pronounced when using padded sequences or variable-length inputs, as described in `this GitHub issue <https://github.com/Lightning-AI/pytorch-lightning/issues/20350>`_.

For more details on this behavior and why it matters, see the `Unsloth blog post <https://unsloth.ai/blog/gradient>`_ and `Hugging Face gradient accumulation guide <https://huggingface.co/blog/gradient_accumulation>`_.
