.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.callbacks.finetuning_scheduler.fts import FinetuningScheduler

.. _finetuning_scheduler:

********************
Finetuning Scheduler
********************

The :class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler` callback enables multi-phase,
scheduled finetuning of foundational models. Gradual unfreezing (i.e. thawing) can help maximize foundational model
knowledge retention while allowing (typically upper layers of) the model to optimally adapt to new tasks during
transfer learning [#]_ [#]_ [#]_ .

:class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler` orchestrates the gradual unfreezing
of models via a finetuning schedule that is either implicitly generated (the default) or explicitly provided by the user
(more computationally efficient). Each finetuning phase proceeds until the configured
:class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping` callback observes the early stopping criteria are
met. A :class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler` training session completes
when the final phase of the schedule has its early stopping criteria met. See
:ref:`Early Stopping<common/early_stopping:Early stopping>` for more details on that callback's configuration.

.. warning:: The :class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler` callback is in beta
    and subject to change.

Basic Example
=============
If no finetuning schedule is user-provided,
:class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler` will generate a
:ref:`default schedule<advanced/finetuning_scheduler:The Default Finetuning Schedule>` and proceed to finetune
according to the generated schedule, using default :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping`
and :class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts_supporters.FTSCheckpoint` callbacks with
``monitor=val_loss``.

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import FinetuningScheduler

    trainer = Trainer(callbacks=[FinetuningScheduler()])


.. _default schedule:

The Default Finetuning Schedule
===============================
Schedule definition is facilitated via
:meth:`~pytorch_lightning.callbacks.finetuning_scheduler.fts_supporters.SchedulingMixin.gen_ft_schedule` which dumps
a default finetuning schedule (by default using a naive, 2-parameters per level heuristic) which can be adjusted as
desired by the user and/or subsequently passed to the callback. Using the default/implicitly generated schedule will
often be less computationally efficient than a user-defined finetuning schedule but can often serve as a
good baseline for subsquent explicit schedule refinement and will marginally outperform many explicit schedules.


.. _specifying schedule:

Specifying a Finetuning Schedule
================================

To specify a finetuning schedule, it's convenient to first generate the default schedule and then alter the
thawed/unfrozen parameter groups associated with each finetuning phase as desired. Finetuning phases are zero-indexed
and executed in ascending order.

1. Generate the default schedule to :paramref:`~pytorch_lightning.trainer.trainer.Trainer.log_dir` with the name
   (:paramref:`~pytorch_lightning.trainer.trainer.lightning_module`.__class__.__name__)_ft_schedule.yaml

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import FinetuningScheduler

    trainer = Trainer(callbacks=[FinetuningScheduler(gen_ft_sched_only=True)])


2. Alter the schedule as desired.

.. container:: sbs-code

    .. rst-class:: sbs-hdr1

        This boring model has four finetuning phases:

    .. rst-class:: sbs-blk1

    .. code-block:: yaml
      :linenos:
      :emphasize-lines: 10

            0:
            - layer.3.bias
            - layer.3.weight
            1:
            - layer.2.bias
            - layer.2.weight
            2:
            - layer.1.bias
            - layer.1.weight
            3:
            - layer.0.bias
            - layer.0.weight

    .. rst-class:: sbs-hdr2

        After removing line 10, three finetuning phases are scheduled:

    .. rst-class:: sbs-blk2

    .. code-block:: yaml
      :linenos:

        0:
        - layer.3.bias
        - layer.3.weight
        1:
        - layer.2.bias
        - layer.2.weight
        2:
        - layer.1.bias
        - layer.1.weight
        - layer.0.bias
        - layer.0.weight


3. Once the finetuning schedule has been altered as desired, pass it to
   :class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler` to commence scheduled training:

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import FinetuningScheduler

    trainer = Trainer(callbacks=[FinetuningScheduler(ft_schedule="/path/to/my/schedule/my_schedule.yaml")])


For a practical end-to-end example of using
:class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler` in implicit versus explicit modes,
see :ref:`scheduled finetuning for SuperGLUE<scheduled-finetuning-superglue>` below.


Resuming Scheduled Finetuning Training Sessions
===============================================

Resumption of scheduled finetuning training is identical to the continuation of
:ref:`other training sessions<common/trainer:trainer>` with the caveat that the provided checkpoint must
have been saved by a :class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler` session.
:class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler` uses
:class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts_supporters.FTSCheckpoint` (an extension of
:class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`) to maintain schedule state with special
metadata.


.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import FinetuningScheduler

    trainer = Trainer(callbacks=[FinetuningScheduler()], resume_from_checkpoint="some/path/to/my_checkpoint.ckpt")

Training will resume at the depth/level of the provided checkpoint according the specified schedule. Schedules can be
altered between training sessions but schedule compatibility is left to the user for maximal flexibility. If executing a
user-defined schedule, typically the same schedule should be provided for the original and resumed training
sessions.


.. tip::

    By default (
    :paramref:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler.restore_best` is ``True``),
    :class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler` will attempt to restore
    the best available checkpoint before finetuning depth transitions.

    .. code-block:: python

        trainer = Trainer(
            callbacks=[FinetuningScheduler(new_incarnation_mode=True)],
            resume_from_checkpoint="some/path/to/my_kth_best_checkpoint.ckpt",
        )

    To handle the edge case wherein one is resuming scheduled finetuning from a non-best checkpoint and the previous
    best checkpoints may not be accessible, setting
    :paramref:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler.new_incarnation_mode` to
    ``True`` as above will re-intialize the checkpoint state with a new best checkpoint at the resumption depth.

Finetuning all the way down!
============================

There are plenty of options for customizing
:class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler`'s behavior, see
:ref:`scheduled finetuning for SuperGLUE<scheduled-finetuning-superglue>` below for examples of composing different
configurations.


.. note::
   Currently, :class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler` only supports
   the following :class:`~pytorch_lightning.plugins.training_type.training_type_plugin.TrainingTypePlugin` s:

   .. hlist::
      :columns: 3

      * :class:`~pytorch_lightning.plugins.training_type.DDPPlugin`
      * :class:`~pytorch_lightning.plugins.training_type.DDPShardedPlugin`
      * :class:`~pytorch_lightning.plugins.training_type.DDPSpawnPlugin`
      * :class:`~pytorch_lightning.plugins.training_type.DDPSpawnShardedPlugin`
      * :class:`~pytorch_lightning.plugins.training_type.DataParallelPlugin`
      * :class:`~pytorch_lightning.plugins.training_type.SingleDevicePlugin`

----------

.. _scheduled-finetuning-superglue:

Example: Scheduled Finetuning For SuperGLUE
===========================================

A demonstration of the scheduled finetuning callback
:class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler` using the
`RTE <https://huggingface.co/datasets/viewer/?dataset=super_glue&config=rte>`_ and
`BoolQ <https://github.com/google-research-datasets/boolean-questions>`_ tasks of the
`SuperGLUE <https://super.gluebenchmark.com/>`_ benchmark and the :ref:`LightningCLI<common/lightning_cli:LightningCLI>`
is available under ./pl_examples/basic_examples/ (depends upon the ``transformers`` and ``datasets`` packages from
Hugging Face)

There are three different demo schedule configurations composed with shared defaults (./config/fts/fts_defaults.yaml)
provided for the default 'rte' task. Note DDP w/ 2 GPUs is the default configuration so ensure you adjust the
configuration files referenced below as desired for other configurations.

.. code-block:: bash

    # Generate a baseline without scheduled finetuning enabled:
    python fts_superglue.py fit --config config/fts/nofts_baseline.yaml

    # Train with the default finetuning schedule:
    python fts_superglue.py fit --config config/fts/fts_implicit.yaml

    # Train with a non-default finetuning schedule:
    python fts_superglue.py fit --config config/fts/fts_explicit.yaml


All three training scenarios use identical configurations with the exception of the provided finetuning schedule. See
the |tensorboard_summ| and table below for a characterization of the relative computational and performance tradeoffs
associated with these :class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler` configurations.
Note that though this example is intended to capture "typical" performance/computational tradeoffs of
:class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler`, substantial variation is expected
among use cases.


.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - | **Example Scenario**
     - | **nofts_baseline**
     - | **fts_implicit**
     - | **fts_explicit**
   * - | Finetuning Schedule
     - None
     - Default
     - User-defined
   * - | RTE Accuracy
       | (``0.69``, ``0.75``, ``0.77``)
     -
        .. raw:: html

            <div style='width:150px;height:auto'>
                <a target="_blank" rel="noopener noreferrer" href="https://tensorboard.dev/experiment/Qy917MVDRlmkx31A895CzA/#scalars&_smoothingWeight=0&runSelectionState=eyJmdHNfaW1wbGljaXQiOmZhbHNlLCJmdHNfZXhwbGljaXQiOmZhbHNlfQ%3D%3D">
                    <img alt="open tensorboard experiment" src="../_static/images/lightning_examples/nofts_baseline.png">
                </a>
            </div>
     -
        .. raw:: html

            <div style='width:150px;height:auto'>
                <a target="_blank" rel="noopener noreferrer" href="https://tensorboard.dev/experiment/Qy917MVDRlmkx31A895CzA/#scalars&_smoothingWeight=0&runSelectionState=eyJmdHNfaW1wbGljaXQiOnRydWUsImZ0c19leHBsaWNpdCI6ZmFsc2UsIm5vZnRzX2Jhc2VsaW5lIjpmYWxzZX0%3D">
                    <img alt="open tensorboard experiment" src="../_static/images/lightning_examples/fts_implicit.png">
                </a>
            </div>
     -
        .. raw:: html

            <div style='width:150px;height:auto'>
                <a target="_blank" rel="noopener noreferrer" href="https://tensorboard.dev/experiment/Qy917MVDRlmkx31A895CzA/#scalars&_smoothingWeight=0&runSelectionState=eyJmdHNfaW1wbGljaXQiOmZhbHNlLCJmdHNfZXhwbGljaXQiOnRydWUsIm5vZnRzX2Jhc2VsaW5lIjpmYWxzZX0%3D">
                    <img alt="open tensorboard experiment" src="../_static/images/lightning_examples/fts_explicit.png">
                </a>
            </div>
   * - | Validation Loss
       | (``0.59``, ``0.50``, ``0.47``)
     -
        .. raw:: html

            <div style='width:150px;height:auto'>
                <a target="_blank" rel="noopener noreferrer" href="https://tensorboard.dev/experiment/Qy917MVDRlmkx31A895CzA/#scalars&_smoothingWeight=0&runSelectionState=eyJmdHNfaW1wbGljaXQiOmZhbHNlLCJmdHNfZXhwbGljaXQiOmZhbHNlfQ%3D%3D">
                    <img alt="open tensorboard experiment" src="../_static/images/lightning_examples/nofts_baseline_loss.png">
                </a>
            </div>
     -
        .. raw:: html

            <div style='width:150px;height:auto'>
                <a target="_blank" rel="noopener noreferrer" href="https://tensorboard.dev/experiment/Qy917MVDRlmkx31A895CzA/#scalars&_smoothingWeight=0&runSelectionState=eyJmdHNfaW1wbGljaXQiOnRydWUsImZ0c19leHBsaWNpdCI6ZmFsc2UsIm5vZnRzX2Jhc2VsaW5lIjpmYWxzZX0%3D">
                    <img alt="open tensorboard experiment" src="../_static/images/lightning_examples/fts_implicit_loss.png">
                </a>
            </div>
     -
        .. raw:: html

            <div style='width:150px;height:auto'>
                <a target="_blank" rel="noopener noreferrer" href="https://tensorboard.dev/experiment/Qy917MVDRlmkx31A895CzA/#scalars&_smoothingWeight=0&runSelectionState=eyJmdHNfaW1wbGljaXQiOmZhbHNlLCJmdHNfZXhwbGljaXQiOnRydWUsIm5vZnRzX2Jhc2VsaW5lIjpmYWxzZX0%3D">
                    <img alt="open tensorboard experiment" src="../_static/images/lightning_examples/fts_explicit_loss.png">
                </a>
            </div>

In summary,
:class:`~pytorch_lightning.callbacks.finetuning_scheduler.fts.FinetuningScheduler` can be used to achieve
non-trivial model performance improvements in both implicit and explicit scheduling contexts at an also non-trivial
computational cost.

.. figure:: ../_static/images/lightning_examples/fts_explicit_loss_anim.gif
   :alt: FinetuningScheduler Explicit Loss Animation
   :width: 300

Footnotes
=========

.. [#] `Howard, J., & Ruder, S. (2018) <https://arxiv.org/pdf/1801.06146.pdf>`_. Fine-tuned Language Models for Text
 Classification. ArXiv, abs/1801.06146.
.. [#] `Chronopoulou, A., Baziotis, C., & Potamianos, A. (2019) <https://arxiv.org/pdf/1902.10547.pdf>`_. An
 embarrassingly simple approach for transfer learning from pretrained language models. arXiv preprint arXiv:1902.10547.
.. [#] `Peters, M. E., Ruder, S., & Smith, N. A. (2019) <https://arxiv.org/pdf/1903.05987.pdf>`_. To tune or not to
 tune? adapting pretrained representations to diverse tasks. arXiv preprint arXiv:1903.05987.

.. seealso::
    - :class:`~pytorch_lightning.trainer.trainer.Trainer`
    - :class:`~pytorch_lightning.callbacks.early_stopping.EarlyStopping`
    - :class:`~pytorch_lightning.callbacks.finetuning.BaseFinetuning`

.. |tensorboard_summ| raw:: html

            <a target="_blank" rel="noopener noreferrer" href="https://tensorboard.dev/experiment/Qy917MVDRlmkx31A895CzA/#scalars&_smoothingWeight=0&runSelectionState=eyJmdHNfZXhwbGljaXQiOnRydWUsImZ0c19pbXBsaWNpdCI6dHJ1ZSwibm9mdHNfYmFzZWxpbmUiOnRydWV9">
            tensorboard experiment summaries
            </a>
