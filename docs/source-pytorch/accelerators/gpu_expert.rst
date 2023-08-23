:orphan:

.. _gpu_expert:

GPU training (Expert)
=====================
**Audience:** Experts creating new scaling techniques such as :ref:`FSDP <fully-sharded-training>` or :ref:`DeepSpeed <deepspeed_advanced>`.

.. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

----

Lightning enables experts focused on researching new ways of optimizing distributed training/inference strategies to create new strategies and plug them into Lightning.

For example, Lightning worked closely with the Microsoft team to develop a :ref:`DeepSpeed <deepspeed_advanced>` integration and with the Facebook (Meta) team to develop a :ref:`FSDP <fully-sharded-training>` integration.


----

.. include:: ../extensions/strategy.rst


----

.. include:: ../advanced/strategy_registry.rst
