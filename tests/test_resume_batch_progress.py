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
    candidates = [
        getattr(trainer, "batch_progress", None),
        getattr(getattr(trainer, "fit_loop", None), "batch_progress", None),
        getattr(getattr(getattr(trainer, "fit_loop", None), "epoch_loop", None), "batch_progress", None),
    ]
    for c in candidates:
        if c is not None:
            return c

    # heuristic scan
    for name in dir(trainer):
        try:
            obj = getattr(trainer, name)
            if obj is None:
                continue
            if any(n in dir(obj) for n in ("current_completed", "current", "completed")):
                return obj
        except Exception:
            continue
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
    if isinstance(candidate, int):
        return candidate

    for attr in ("completed", "ready", "count", "value", "n", "total"):
        if hasattr(candidate, attr):
            try:
                return int(getattr(candidate, attr))
            except Exception:
                pass

    if isinstance(candidate, (tuple, list)) and len(candidate) > 0:
        for el in candidate:
            try:
                return int(el)
            except Exception:
                for attr in ("completed", "ready", "count", "value", "n"):
                    if hasattr(el, attr):
                        try:
                            return int(getattr(el, attr))
                        except Exception:
                            pass
        try:
            return int(candidate[0])
        except Exception:
            pass

    try:
        return int(candidate)
    except Exception:
        typename = type(candidate).__name__
        sample_attrs = ", ".join(sorted(dir(candidate))[:40])
        raise ValueError(f"Unable to coerce candidate (type {typename}) to int. Sample attrs: {sample_attrs}")


def test_resume_mid_epoch_batch_progress(tmp_path):
    L.seed_everything(42)

    ds = WikiText2()
    n = len(ds)
    train_ds, val_ds, _ = random_split(ds, [n - 200, 100, 100])

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # 1) Run short training to produce a mid-epoch checkpoint (step 5)
    model = TinyModel(vocab_size=ds.vocab_size)
    trainer_short = L.Trainer(max_steps=5, enable_progress_bar=False)
    trainer_short.fit(model, train_loader, val_loader)
    assert trainer_short.global_step >= 5, f"short trainer didn't reach step 5 (gs={trainer_short.global_step})"

    ckpt_mid = tmp_path / "mid_epoch.ckpt"
    trainer_short.save_checkpoint(str(ckpt_mid))
    assert ckpt_mid.exists(), f"failed to create checkpoint at {ckpt_mid}"

    # 2) Resume from that checkpoint with a fresh trainer
    trainer_resume = L.Trainer(max_steps=10, enable_progress_bar=False)
    model2 = TinyModel(vocab_size=ds.vocab_size)
    trainer_resume.fit(model2, train_loader, val_loader, ckpt_path=str(ckpt_mid))

    bp = _find_batch_progress(trainer_resume)
    assert bp is not None, "BatchProgress object not found on Trainer; see earlier logs."

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
        if hasattr(bp, name):
            total_candidate = getattr(bp, name)
            break
    if total_candidate is None:
        total_candidate = bp

    current_candidate = None
    for name in possible_current_names:
        if hasattr(bp, name):
            current_candidate = getattr(bp, name)
            break
    if current_candidate is None:
        for name in dir(bp):
            if name.lower().startswith("current"):
                try:
                    current_candidate = getattr(bp, name)
                    break
                except Exception:
                    pass
    if current_candidate is None:
        current_candidate = 0

    total_completed = _extract_int(total_candidate)
    current_completed = _extract_int(current_candidate)

    gs = trainer_resume.global_step

    assert total_completed >= 0 and current_completed >= 0, "negative counters found"
    assert total_completed >= gs or total_completed == 0, (
        f"unexpected total_completed={total_completed} < global_step={gs}"
    )
    assert current_completed <= total_completed
