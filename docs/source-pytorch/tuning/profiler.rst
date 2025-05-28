.. _profiler:

#############################
Find bottlenecks in your code
#############################

.. warning::

    **Do not wrap** ``Trainer.fit()``, ``Trainer.validate()``, or other Trainer methods
    inside a manual ``torch.profiler.profile`` context manager.  
    This will cause unexpected crashes and cryptic errors due to incompatibility between
    PyTorch Profiler's context management and Lightning's internal training loop.
    Instead, always use the ``profiler`` argument in the ``Trainer`` constructor.

    Example (correct usage):

    .. code-block:: python

        import pytorch_lightning as pl

        trainer = pl.Trainer(
            profiler="pytorch",  # <- This enables built-in profiling safely!
            ...
        )
        trainer.fit(model, train_dataloaders=...)

    **References:**
      - https://github.com/pytorch/pytorch/issues/88472
      - https://github.com/Lightning-AI/lightning/issues/16958

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Basic
   :description: Learn to find bottlenecks in the training loop.
   :col_css: col-md-3
   :button_link: profiler_basic.html
   :height: 150
   :tag: basic

.. displayitem::
   :header: Intermediate
   :description: Learn to find bottlenecks in PyTorch operations.
   :col_css: col-md-3
   :button_link: profiler_intermediate.html
   :height: 150
   :tag: intermediate

.. displayitem::
   :header: Advanced
   :description: Learn to profile TPU code.
   :col_css: col-md-3
   :button_link: profiler_advanced.html
   :height: 150
   :tag: advanced

.. displayitem::
   :header: Expert
   :description: Learn to build your own profiler or profile custom pieces of code
   :col_css: col-md-3
   :button_link: profiler_expert.html
   :height: 150
   :tag: expert

.. raw:: html

        </div>
    </div>
