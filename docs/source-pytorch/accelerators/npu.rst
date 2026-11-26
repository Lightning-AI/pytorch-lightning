.. _npu:

Accelerator: Ascend NPU training
================================

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. Add callout items below this line

.. displayitem::
   :header: Prepare your code (Optional)
   :description: Prepare your code to run on any hardware
   :col_css: col-md-4
   :button_link: accelerator_prepare.html
   :height: 150
   :tag: basic

.. displayitem::
   :header: Install & Quick Start
   :description: Install the Ascend NPU stack and run your first training.
   :col_css: col-md-4
   :button_link: ../accelerators/npu.html
   :height: 150
   :tag: basic

.. raw:: html

        </div>
    </div>

This guide documents the **Ascend NPU** accelerator support.
PyTorch Lightning has been adapted to run on Huawei Ascend NPUs (910B series)
while fully preserving the original CUDA/GPU code paths.

.. note::

   NPU support is built on ``torch_npu`` + ``torch_npu.contrib.transfer_to_npu``
   (runtime mechanical migration) plus a native ``NPUAccelerator`` platform integration
   (registry / auto-probe / backend mapping). All CUDA branches, enums, and imports from
   upstream Lightning are preserved untouched.

******************************
Installation (NPU environment)
******************************

The NPU runtime is *additive*: you keep your existing Lightning install and add the
Ascend stack on top. **Do not uninstall** the packages that ``pip install lightning``
brings in (Lightning metadata is hardware-neutral — no ``+cu`` pins, no ``nvidia-*``
hard deps). The only component you must replace is **torch itself**.

Prerequisites / software stack (verified on aarch64 Linux + Python 3.11, 8× 910B3, CANN 9.2.0;
on x86_64 use the CPU ``--index-url`` shown below):

.. code-block:: bash

    # 1. NPU driver / firmware (Ascend 910B3) — from the Huawei Ascend support site
    # 2. CANN toolkit 9.2.0 — https://www.hiascend.com/software/cann
    #    (driver / firmware must match CANN 9.2.0 and torch_npu 2.10.0.post2)
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    # 3. torch 2.10.0 CPU build + torch_npu 2.10.0.post2  (versions MUST match exactly)
    python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu  # CPU wheel: avoids pulling CUDA torch + nvidia-* on x86_64
    python -m pip install torch_npu==2.10.0.post2

    # 4. operator extensions (as required by your workload)
    python -m pip install triton_ascend==3.2.2

    # 5. DeepSpeed — use the Ascend NPU build (separate install, NOT the PyPI extra)
    #    0.20.0 is the Ascend build, deliberately chosen above the declared ``>=0.16.0,<0.17.0``
    #    range; do not mix with the PyPI release or the ``[strategies]`` extra
    python -m pip install deepspeed==0.20.0   # Ascend NPU build (npu_accelerator + op_builder/npu)

    # 6. jsonargparse — required by LightningCLI (not pulled in once [pytorch-extra] is skipped)
    python -m pip install "jsonargparse[signatures,jsonnet]>=4.39,<4.51"

    # 7. Lightning — version range satisfied, will not overwrite torch
    python -m pip install lightning

.. tip::

   To skip most of the manual setup, use the official Ascend container image
   ``quay.io/ascend/torch-npu`` (pre-installed CANN + PyTorch + torch_npu) and only
   install Lightning on top.

.. warning::

   - Do **not** install the CUDA-ecosystem extras: ``lightning[strategies]`` / ``lightning[all]`` /
     ``lightning[pytorch-extra]`` pull in CUDA builds from PyPI. The ``lightning`` meta package merges
     the ``pytorch`` and ``fabric`` extra sets: ``lightning[strategies]`` pulls ``deepspeed`` (pytorch)
     and ``bitsandbytes`` (fabric); ``lightning[pytorch-extra]`` pulls ``bitsandbytes``, ``jsonargparse``,
     etc. On NPU install the needed pieces separately as Ascend builds.
   - ``bitsandbytes`` quantization is **not enabled** on NPU (CUDA branch preserved, NPU raises a clear
     ``MisconfigurationException``). ``transformer-engine`` FP8 is NVIDIA-only and not enabled on NPU.

Environment variables for NPU runs:

.. code-block:: bash

    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # device mask (like CUDA_VISIBLE_DEVICES)
    export HCCL_NPU_SOCKET_PORT_RANGE=17000-17099      # avoid HCCL default port 16666 conflicts (single host multi-process)
    export MASTER_PORT=25998                            # torchrun master port isolation
    export TORCH_NPU_DEVICE_CAPABILITY=9.0              # only needed when using torch.compile on NPU (dynamo TMA check)

Smoke test
----------

Run these before training to confirm the stack is wired up correctly:

.. code-block:: bash

    npu-smi info                                                     # driver / device visible
    python -c "import torch; print(torch.npu.is_available())"        # should print True
    python -c "import torch; print(torch.randn(2, 2).npu().sum())"    # minimal tensor op on NPU
    python -c "import lightning.pytorch; print(lightning.pytorch.__version__)"  # imports OK

*****************************************
Quick start (training on NPU)
*****************************************

The framework auto-detects NPU: ``accelerator="npu"`` selects the ``NPUAccelerator``,
and ``accelerator="auto"`` probes ``tpu -> mps -> npu -> cuda -> cpu`` (NPU preferred
over CUDA on Ascend machines).

.. code-block:: python

    import torch
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader, TensorDataset

    class M(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(8, 2)

        def training_step(self, batch, batch_idx):
            x, y = batch
            return torch.nn.functional.cross_entropy(self.l(x), y)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    x, y = torch.randn(16, 8), torch.randint(0, 2, (16,))
    pl.Trainer(
        accelerator="npu", devices=1, max_epochs=1,
        enable_checkpointing=False, logger=False,
    ).fit(M(), DataLoader(TensorDataset(x, y), batch_size=8))

Distributed training on NPU (HCCL backend):

.. code-block:: python

    # 2-card DDP
    pl.Trainer(accelerator="npu", strategy="ddp", devices=2, max_epochs=1).fit(model, dataloader)

    # 8-card DDP
    pl.Trainer(accelerator="npu", strategy="ddp", devices=8, max_epochs=1).fit(model, dataloader)

    # FSDP / DeepSpeed (Ascend build) / ModelParallel also verified on NPU with the hccl backend.

The process-group backend maps ``npu -> hccl`` automatically
(``torch.distributed.Backend.default_device_backend_map["npu"] == "hccl"``).

***********************************************
``transfer_to_npu`` automatic migration
***********************************************

Mechanical ``cuda -> npu`` replacement points (``torch.cuda.*``, ``torch.device("cuda")``,
``nccl``, etc.) are handled **at runtime** by ``torch_npu.contrib.transfer_to_npu`` rather than
being edited call-site by call-site. This minimizes the diff while keeping the CUDA
calls untouched.

**Activation is double-gated** (see ``src/lightning/fabric/accelerators/npu.py``): it is only
triggered when (a) ``import torch_npu`` succeeds **and** (b) a real NPU device is visible
(``torch.npu.is_available()``). Importing ``lightning.pytorch`` / ``lightning.fabric`` on a
pure-CUDA machine never activates the migration — the CUDA path is unaffected.

Verified working scope (8× 910B3):

- ``Trainer(accelerator="npu", ...)`` single card, DDP 2/8 cards, FSDP, DeepSpeed ZeRO-2 (Ascend build),
  ModelParallel (tensor parallel), AMP ``16-mixed`` / ``bf16-mixed``, ``torch.compile`` (``backend="npu"``),
  Checkpoint save/resume, Tuner, Profiler, deterministic/seed, LightningCLI, Fabric, ServableModule.
- ``Trainer.predict()`` batch inference.
- HTTP inference serving (``ServableModuleValidator``, ``fastapi`` + ``uvicorn``).
- Demo training with ``LightningLSTM`` and ``LightningTransformer``.

Known boundaries:

- ``bitsandbytes`` quantization: **not enabled** on NPU (clear error, CUDA branch preserved).
- ``transformer-engine`` FP8: NVIDIA-only, **not enabled** on NPU.
- FSDP ``HYBRID_SHARD`` (2D device mesh with ``mesh_dim_names``): HCCL does not support ``split_group``
  (see the ``split_group`` limitation in ``src/lightning/pytorch/strategies/model_parallel.py``).
- ``transfer_to_npu`` disables ``torch.jit.script`` / ``torch.jit.script_method``.
- After ``Trainer.fit`` the module is moved back to CPU by ``Strategy.teardown`` (same as CUDA); assert
  device during training callbacks, not after ``fit``.
- Some upstream operators (e.g. ``aten::_transformer_encoder_layer_fwd``) may fall back to CPU on NPU —
  a performance concern, not a correctness error.

.. note::

   All CUDA/GPU branches from upstream Lightning are preserved. NPU support is added
   alongside them (new ``npu`` branches / registry entries), never by deleting GPU code.
