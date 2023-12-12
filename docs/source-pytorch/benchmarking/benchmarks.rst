:orphan:

Benchmark performance vs. vanilla PyTorch
=========================================

In this section we set grounds for comparison between vanilla PyTorch and PT Lightning for most common scenarios.

Time comparison
---------------

We have set regular benchmarking against PyTorch vanilla training loop on with RNN and simple MNIST classifier as per of out CI.
In average for simple MNIST CNN classifier we are only about 0.06s slower per epoch, see detail chart below.

.. figure:: ../_static/images/benchmarks/figure-parity-times.png
   :alt: Speed parity to vanilla PT, created on 2020-12-16
   :width: 500


Learn more about reproducible benchmarking from the `PyTorch Reproducibility Guide <https://pytorch.org/docs/stable/notes/randomness.html>`__.


----

Find performance bottlenecks
=============================

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Find bottlenecks in your models
   :description: Benchmark your own Lightning models
   :button_link: ../tuning/profiler.html
   :col_css: col-md-3
   :height: 180
   :tag: basic

.. raw:: html

        </div>
    </div>
