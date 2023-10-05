.. list-table:: adv. user 2.0
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - used the ``torchdistx`` package and integration in Trainer
     - materialize the model weights manually, or follow our guide for initializing large models
     - `PR17995`_

   * - defined ``def training_step(self, dataloader_iter, batch_idx)`` in LightningModule
     - remove ``batch_idx`` from the signature and expect ``dataloader_iter`` to return a triplet ``(batch, batch_idx, dataloader_idx)``
     - `PR18390`_

   * - defined ``def validation_step(self, dataloader_iter, batch_idx)`` in LightningModule
     - remove ``batch_idx`` from the signature and expect ``dataloader_iter`` to return a triplet ``(batch, batch_idx, dataloader_idx)``
     - `PR18390`_

   * - defined ``def test_step(self, dataloader_iter, batch_idx)`` in LightningModule
     - remove ``batch_idx`` from the signature and expect ``dataloader_iter`` to return a triplet ``(batch, batch_idx, dataloader_idx)``
     - `PR18390`_

   * - defined ``def predict_step(self, dataloader_iter, batch_idx)`` in LightningModule
     - remove ``batch_idx`` from the signature and expect ``dataloader_iter`` to return a triplet ``(batch, batch_idx, dataloader_idx)``
     - `PR18390`_

   * - used ``batch = next(dataloader_iter)`` in LightningModule ``*_step`` hooks
     - use ``batch, batch_idx, dataloader_idx = next(dataloader_iter)``
     - `PR18390`_


.. _pr17995: https://github.com/Lightning-AI/lightning/pull/17995
.. _pr18390: https://github.com/Lightning-AI/lightning/pull/18390
