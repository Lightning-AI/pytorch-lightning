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
