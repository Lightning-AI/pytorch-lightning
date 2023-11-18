.. list-table:: devel 2.0
   :widths: 40 40 20
   :header-rows: 1

   * - If
     - Then
     - Ref

   * - used the ``XLAStrategy.is_distributed`` property
     - it was removed because it was always True
     - `PR17381`_

   * - used the ``SingleTPUStrategy.is_distributed`` property
     - it was removed because it was always False
     - `PR17381`_


.. _pr17381: https://github.com/Lightning-AI/lightning/pull/17381
