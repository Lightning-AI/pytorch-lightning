.. _profiler:

#############################
Find bottlenecks in your code
#############################

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. warning::

    Do **not** wrap ``Trainer.fit()``, ``Trainer.validate()``, or similar Trainer methods inside a manual ``torch.profiler.profile`` context manager.
    This will cause unexpected crashes and cryptic errors due to incompatibility between PyTorch Profiler's context and Lightning's training loop.
    Instead, use the ``profiler`` argument of the ``Trainer``:

    .. code-block:: python

        trainer = pl.Trainer(
            profiler="pytorch",  # This is the correct and supported way
            ...
        )

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
