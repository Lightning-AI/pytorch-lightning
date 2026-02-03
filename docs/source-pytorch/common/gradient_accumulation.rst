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

CrossEntropyLoss Normalization for Gradient Accumulation
========================================================

When using ``CrossEntropyLoss`` with gradient accumulation on variable-length sequences (e.g., language modeling with padding),
you can override the default normalization by returning a ``"normalize"`` key from ``training_step``.

.. code-block:: python

    class LanguageModel(LightningModule):
        def training_step(self, batch, batch_idx):
            # Compute total valid tokens across all batches in accumulation window
            # (only at the start of each window)
            if batch_idx % self.trainer.accumulate_grad_batches == 0:
                total_tokens = sum_valid_tokens_in_next_N_batches()
                self.normalize_value = total_tokens

            logits = self.model(batch["input_ids"])
            labels = batch["labels"]

            # Use reduction='sum' and return normalize value
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   labels.view(-1),
                                   ignore_index=-100,
                                   reduction='sum')

            return {"loss": loss, "normalize": self.normalize_value}

The ``"normalize"`` value should be the total count of valid elements across all micro-batches in the accumulation window.
Use ``reduction='sum'`` in your loss function and return the total token count as the normalization factor.
See the `Unsloth blog post <https://unsloth.ai/blog/gradient>`_ for details.
