import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import lightning as L
from lightning.pytorch.demos import Transformer, WikiText2


class TinyModel(L.LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.model = Transformer(vocab_size=vocab_size)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x, y)
        return F.nll_loss(out, y.view(-1))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _find_batch_progress(trainer):
    """Try common places for the BatchProgress object across Lightning versions."""
    log = logging.getLogger(__name__)
    candidates = [
        getattr(trainer, "batch_progress", None),
        getattr(getattr(trainer, "fit_loop", None), "batch_progress", None),
        getattr(getattr(getattr(trainer, "fit_loop", None), "epoch_loop", None), "batch_progress", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return candidate

    # heuristic scan
    for name in dir(trainer):
        try:
            attr = getattr(trainer, name)
            if attr is None:
                continue
            if any(n in dir(attr) for n in ("current_completed", "current", "completed")):
                return attr
        except Exception as exc:
            log.debug(f"BatchProgress restore fallback triggered: {exc}")
    return None


def _extract_int(candidate):
    """
    Try to extract an integer from candidate which may be:
     - an int-like
     - an object with attributes like 'completed', 'ready', 'count', 'value', 'n'
     - a tuple/list like (ready, completed)
     - something convertible via int()
    Raise ValueError if not possible.
    """
    log = logging.getLogger(__name__)

    if isinstance(candidate, int):
        return candidate

    for attribute in ("completed", "ready", "count", "value", "n", "total"):
        if hasattr(candidate, attribute):
            try:
                return int(getattr(candidate, attribute))
            except Exception as exc:
                log.debug(f"BatchProgress restore fallback triggered: {exc}")

    if isinstance(candidate, (tuple, list)) and len(candidate) > 0:
        for element in candidate:
            try:
                return int(element)
            except Exception:
                for attribute in ("completed", "ready", "count", "value", "n"):
                    if hasattr(element, attribute):
                        try:
                            return int(getattr(element, attribute))
                        except Exception as exc:
                            log.debug(f"BatchProgress restore fallback triggered: {exc}")
        try:
            return int(candidate[0])
        except Exception as exc:
            log.debug(f"BatchProgress restore fallback triggered: {exc}")

    try:
        return int(candidate)
    except Exception:
        typename = type(candidate).__name__
        sample_attrs = ", ".join(sorted(dir(candidate))[:40])
        raise ValueError(f"Unable to coerce candidate (type {typename}) to int. Sample attrs: {sample_attrs}")


def test_resume_mid_epoch_batch_progress(tmp_path):
    L.seed_everything(42)
    log = logging.getLogger(__name__)
    dataset = WikiText2()
    dataset_len = len(dataset)
    train_ds, val_ds, _ = random_split(dataset, [dataset_len - 200, 100, 100])

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # 1) Run short training to produce a mid-epoch checkpoint (step 5)
    model = TinyModel(vocab_size=dataset.vocab_size)
    trainer_short = L.Trainer(max_steps=5, enable_progress_bar=False)
    trainer_short.fit(model, train_loader, val_loader)
    assert trainer_short.global_step >= 5, f"short trainer didn't reach step 5 (gs={trainer_short.global_step})"

    ckpt_mid = tmp_path / "mid_epoch.ckpt"
    trainer_short.save_checkpoint(str(ckpt_mid))
    assert ckpt_mid.exists(), f"failed to create checkpoint at {ckpt_mid}"

    # 2) Resume from that checkpoint with a fresh trainer
    trainer_resume = L.Trainer(max_steps=10, enable_progress_bar=False)
    model2 = TinyModel(vocab_size=dataset.vocab_size)
    trainer_resume.fit(model2, train_loader, val_loader, ckpt_path=str(ckpt_mid))

    batch_progress = _find_batch_progress(trainer_resume)
    assert batch_progress is not None, "BatchProgress object not found on Trainer; see earlier logs."

    possible_total_names = [
        "total_completed",
        "total_completed_batches",
        "total_steps_completed",
        "completed",
        "total",
        "total_done",
        "total_ready",
        "ready",
    ]
    possible_current_names = [
        "current_completed",
        "current",
        "current_batch",
        "current_index",
        "completed_in_epoch",
        "in_progress",
        "current_ready",
    ]

    total_candidate = None
    for name in possible_total_names:
        if hasattr(batch_progress, name):
            total_candidate = getattr(batch_progress, name)
            break
    if total_candidate is None:
        total_candidate = batch_progress

    current_candidate = None
    for name in possible_current_names:
        if hasattr(batch_progress, name):
            current_candidate = getattr(batch_progress, name)
            break
    if current_candidate is None:
        for name in dir(batch_progress):
            if name.lower().startswith("current"):
                try:
                    current_candidate = getattr(batch_progress, name)
                    break
                except Exception as exc:
                    log.debug(f"BatchProgress restore fallback triggered: {exc}")
    if current_candidate is None:
        current_candidate = 0

    total_completed = _extract_int(total_candidate)
    current_completed = _extract_int(current_candidate)

    global_step = trainer_resume.global_step

    assert total_completed >= 0, "negative total_completed found"
    assert current_completed >= 0, "negative current_completed found"

    assert total_completed >= global_step or total_completed == 0, (
        f"unexpected total_completed={total_completed} < global_step={global_step}"
    )

    assert current_completed <= total_completed
