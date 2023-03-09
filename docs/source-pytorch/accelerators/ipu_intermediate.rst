:orphan:

.. _ipu_intermediate:

Accelerator: IPU training
=========================
**Audience:** IPU users looking to increase performance via mixed precision and analysis tools.

.. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

----

Mixed precision & 16 bit precision
----------------------------------

Lightning also supports training in mixed precision with IPUs.
By default, IPU training will use 32-bit precision. To enable mixed precision,
set the precision flag.

.. note::
    Currently there is no dynamic scaling of the loss with mixed precision training.

.. code-block:: python

    import lightning.pytorch as pl

    model = MyLightningModule()
    trainer = pl.Trainer(accelerator="ipu", devices=8, precision=16)
    trainer.fit(model)

You can also use pure 16-bit training, where the weights are also in 16-bit precision.

.. code-block:: python

    import lightning.pytorch as pl
    from lightning.pytorch.strategies import IPUStrategy

    model = MyLightningModule()
    model = model.half()
    trainer = pl.Trainer(accelerator="ipu", devices=8, precision=16)
    trainer.fit(model)

----

PopVision Graph Analyser
------------------------

.. figure:: ../_static/images/accelerator/ipus/profiler.png
   :alt: PopVision Graph Analyser
   :width: 500

Lightning supports integration with the `PopVision Graph Analyser Tool <https://docs.graphcore.ai/projects/graph-analyser-userguide/en/latest/>`__. This helps to look at utilization of IPU devices and provides helpful metrics during the lifecycle of your trainer. Once you have gained access, The PopVision Graph Analyser Tool can be downloaded via the `GraphCore download website <https://downloads.graphcore.ai/>`__.

Lightning supports dumping all reports to a directory to open using the tool.

.. code-block:: python

    import lightning.pytorch as pl
    from lightning.pytorch.strategies import IPUStrategy

    model = MyLightningModule()
    trainer = pl.Trainer(accelerator="ipu", devices=8, strategy=IPUStrategy(autoreport_dir="report_dir/"))
    trainer.fit(model)

This will dump all reports to ``report_dir/`` which can then be opened using the Graph Analyser Tool, see `Opening Reports <https://docs.graphcore.ai/projects/graph-analyser-userguide/en/latest/opening-reports.html>`__.
