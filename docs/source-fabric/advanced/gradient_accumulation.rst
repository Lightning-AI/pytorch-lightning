###############################
Efficient Gradient Accumulation
###############################

Gradient accumulation works the same way with Fabric as in PyTorch.
You are in control of which model accumulates and at what frequency:

.. code-block:: python

    for iteration, batch in enumerate(dataloader):
        # Accumulate gradient 8 batches at a time
        is_accumulating = iteration % 8 != 0

        output = model(input)
        loss = ...

        # .backward() accumulates when .zero_grad() wasn't called
        fabric.backward(loss)
        ...

        if not is_accumulating:
            # Step the optimizer after the accumulation phase is over
            optimizer.step()
            optimizer.zero_grad()


However, in a distributed setting, for example, when training across multiple GPUs or machines, doing it this way can significantly slow down your training loop.
To optimize this code, we should skip the synchronization in ``.backward()`` during the accumulation phase.
We only need to synchronize the gradients when the accumulation phase is over!
This can be achieved by adding the :meth:`~lightning.fabric.fabric.Fabric.no_backward_sync` context manager over the :meth:`~lightning.fabric.fabric.Fabric.backward` call:

.. code-block:: diff

      for iteration, batch in enumerate(dataloader):

          # Accumulate gradient 8 batches at a time
          is_accumulating = iteration % 8 != 0

    +     with fabric.no_backward_sync(model, enabled=is_accumulating):
              output = model(input)
              loss = ...

              # .backward() accumulates when .zero_grad() wasn't called
              fabric.backward(loss)

          ...

          if not is_accumulating:
              # Step the optimizer after accumulation phase is over
              optimizer.step()
              optimizer.zero_grad()


For those strategies that don't support it, a warning is emitted. For single-device strategies, it is a no-op.
Both the model's ``.forward()`` and the ``fabric.backward()`` call need to run under this context.
