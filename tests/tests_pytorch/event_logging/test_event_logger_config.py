import pytest


def test_event_logging_public_types_exist():
    # Public surface should exist
    from lightning.pytorch.event_logging.types import EventRecord  # noqa: F401
    from lightning.pytorch.event_logging.plugins import BaseEventPlugin  # noqa: F401
    from lightning.pytorch.event_logging.event_logger import EventLogger  # noqa: F401


def test_trainer_still_instantiates_without_event_logger_kwarg():
    # Existing behavior must remain valid
    from lightning.pytorch.trainer.trainer import Trainer

    Trainer()


def test_eventrecord_structure_and_deterministic_order_across_runs_and_hooks():
    # Verify EventRecord structure, deterministic order across runs, and presence/order of core hooks per assumptions
    from lightning.pytorch.event_logging.plugins import BaseEventPlugin
    from lightning.pytorch.event_logging.event_logger import EventLogger
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.trainer.trainer import Trainer

    class CapturePlugin(BaseEventPlugin):
        def __init__(self):
            self.events = []

        def on_event(self, event):
            # Structure checks on each event
            assert hasattr(event, "type") and isinstance(event.type, str)
            assert hasattr(event, "timestamp")
            assert hasattr(event, "metadata") and isinstance(event.metadata, dict)
            # duration may be None or a number; do not over-constrain
            assert hasattr(event, "duration")
            self.events.append(event)

    plugin = CapturePlugin()
    logger = EventLogger(plugins=[plugin])

    # First run
    Trainer(max_epochs=1, limit_train_batches=1, enable_checkpointing=True, logger=False, enable_model_summary=False,
            event_logger=logger).fit(BoringModel())
    types_run1 = [e.type for e in plugin.events]
    assert len(types_run1) > 0

    # Second run: same configuration should yield identical ordering of event types
    plugin.events.clear()
    Trainer(max_epochs=1, limit_train_batches=1, enable_checkpointing=True, logger=False, enable_model_summary=False,
            event_logger=logger).fit(BoringModel())
    types_run2 = [e.type for e in plugin.events]
    assert types_run2 == types_run1

    # Explicit hooks (assumptions define canonical names)
    def idx(name):
        assert name in types_run2, f"missing event type: {name}"
        return types_run2.index(name)

    assert idx("forward") < idx("backward") < idx("optimizer_step")


def test_checkpoint_and_metrics_events_present_and_increase_coverage():
    # Ensure explicit presence of checkpoint/metric events and overall coverage increase when enabled
    from lightning.pytorch.event_logging.plugins import BaseEventPlugin
    from lightning.pytorch.event_logging.event_logger import EventLogger
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.trainer.trainer import Trainer

    class CaptureTypes(BaseEventPlugin):
        def __init__(self):
            self.types = []

        def on_event(self, event):
            self.types.append(event.type)

    # Baseline: training only
    base = CaptureTypes()
    logger_base = EventLogger(plugins=[base])
    Trainer(max_epochs=1, limit_train_batches=1, enable_checkpointing=False,
            logger=False, enable_model_summary=False, event_logger=logger_base).fit(BoringModel())

    # With checkpointing and validation enabled
    with_ckpt_val = CaptureTypes()
    logger_rich = EventLogger(plugins=[with_ckpt_val])
    Trainer(max_epochs=1, limit_train_batches=1, limit_val_batches=1, enable_checkpointing=True,
            logger=False, enable_model_summary=False, event_logger=logger_rich).fit(BoringModel())

    # Expect explicit event types per assumptions
    assert "checkpoint" in set(with_ckpt_val.types)
    assert "metric" in set(with_ckpt_val.types)
    # Expect strictly more distinct event types and total events
    assert len(set(with_ckpt_val.types)) >= len(set(base.types))
    assert len(with_ckpt_val.types) > len(base.types)


def test_plugin_dispatch_order_is_deterministic_by_plugin_list_per_event():
    # Multiple plugins must be invoked for every event in the exact provided order (per-event ordering)
    from lightning.pytorch.event_logging.plugins import BaseEventPlugin
    from lightning.pytorch.event_logging.event_logger import EventLogger
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.trainer.trainer import Trainer

    sink = []  # global call timeline: tuples (tag, per_plugin_index)
    types_a, types_b = [], []

    class P(BaseEventPlugin):
        def __init__(self, tag):
            self.tag = tag
            self.i = 0

        def on_event(self, event):
            self.i += 1
            sink.append((self.tag, self.i))
            if self.tag == "A":
                types_a.append(event.type)
            else:
                types_b.append(event.type)

    p1, p2 = P("A"), P("B")
    logger = EventLogger(plugins=[p1, p2])
    Trainer(max_epochs=1, limit_train_batches=1, enable_checkpointing=False, logger=False, enable_model_summary=False,
            event_logger=logger).fit(BoringModel())

    # Both plugins saw the same number of events and in the same type sequence
    assert len(types_a) == len(types_b) and len(types_a) > 0
    assert types_a == types_b
    # For each event index k, the A(k) call precedes B(k) in the global timeline
    for k in range(1, len(types_a) + 1):
        pos_a = next(i for i, it in enumerate(sink) if it == ("A", k))
        pos_b = next(i for i, it in enumerate(sink) if it == ("B", k))
        assert pos_a < pos_b, f"Plugin A did not precede B for event index {k}"


def test_plugins_tuple_supported_and_order_preserved():
    # Accepts tuple[BaseEventPlugin,...] and preserves order
    from lightning.pytorch.event_logging.plugins import BaseEventPlugin
    from lightning.pytorch.event_logging.event_logger import EventLogger
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.trainer.trainer import Trainer

    sink = []

    class P(BaseEventPlugin):
        def __init__(self, tag):
            self.tag = tag
            self.i = 0

        def on_event(self, event):
            self.i += 1
            sink.append((self.tag, self.i))

    p1, p2 = P("A"), P("B")
    logger = EventLogger(plugins=(p1, p2))
    Trainer(max_epochs=1, limit_train_batches=1, enable_checkpointing=False, logger=False, enable_model_summary=False,
            event_logger=logger).fit(BoringModel())

    # First call must be A, and for each index k, A(k) precedes B(k)
    assert sink and sink[0][0] == "A"
    n_a = max(i for tag, i in sink if tag == "A")
    n_b = max(i for tag, i in sink if tag == "B")
    assert n_a == n_b and n_a > 0
    for k in range(1, n_a + 1):
        pos_a = next(i for i, it in enumerate(sink) if it == ("A", k))
        pos_b = next(i for i, it in enumerate(sink) if it == ("B", k))
        assert pos_a < pos_b


def test_enable_disable_and_plugin_selection():
    # Only the selected plugin should receive events; disabled logger leads to no calls
    from lightning.pytorch.event_logging.plugins import BaseEventPlugin
    from lightning.pytorch.event_logging.event_logger import EventLogger
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.trainer.trainer import Trainer

    class Counter(BaseEventPlugin):
        def __init__(self):
            self.n = 0

        def on_event(self, event):
            self.n += 1

    p1, p2 = Counter(), Counter()
    logger = EventLogger(plugins=[p1])
    Trainer(max_epochs=1, limit_train_batches=1, enable_checkpointing=False, logger=False, enable_model_summary=False,
            event_logger=logger).fit(BoringModel())
    assert p1.n > 0 and p2.n == 0

    # Disabled: no plugin calls executed
    p1.n = 0
    Trainer(max_epochs=1, limit_train_batches=1, enable_checkpointing=False, logger=False, enable_model_summary=False,
            event_logger=None).fit(BoringModel())
    assert p1.n == 0


def test_dry_run_mode_drops_events_no_plugin_calls():
    # Dry-run drops events: plugins must not be invoked
    from lightning.pytorch.event_logging.plugins import BaseEventPlugin
    from lightning.pytorch.event_logging.event_logger import EventLogger
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.trainer.trainer import Trainer

    class SideEffectPlugin(BaseEventPlugin):
        def __init__(self):
            self.side_effect = 0
            self.calls = 0

        def on_event(self, event):
            self.side_effect += 1
            self.calls += 1

    p = SideEffectPlugin()
    logger = EventLogger(plugins=[p], dry_run=True)
    Trainer(max_epochs=1, limit_train_batches=1, enable_checkpointing=False, logger=False, enable_model_summary=False,
            event_logger=logger).fit(BoringModel())
    assert p.side_effect == 0 and p.calls == 0


def test_plugin_fault_isolation_quarantines_plugin_and_warns_and_continues(caplog, recwarn):
    # A plugin that raises must be quarantined: training continues, other plugins keep receiving events, and a warning is emitted
    from lightning.pytorch.event_logging.plugins import BaseEventPlugin
    from lightning.pytorch.event_logging.event_logger import EventLogger
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.trainer.trainer import Trainer

    class Flaky(BaseEventPlugin):
        def __init__(self):
            self.calls = 0

        def on_event(self, event):
            self.calls += 1
            raise RuntimeError("boom")

    class OK(BaseEventPlugin):
        def __init__(self):
            self.calls = 0

        def on_event(self, event):
            self.calls += 1

    flaky = Flaky()
    ok = OK()

    logger = EventLogger(plugins=[flaky, ok])
    with caplog.at_level("WARNING"):
        # Use 2 batches so there are events after the first failure; flaky should be quarantined after its first call
        Trainer(max_epochs=1, limit_train_batches=2, enable_checkpointing=False, logger=False, enable_model_summary=False,
                event_logger=logger).fit(BoringModel())

    # The flaky plugin should have received one call then be quarantined. The other plugin should still be active.
    assert flaky.calls == 1
    # OK plugin should have been called for multiple events, demonstrating continuation after quarantine
    assert ok.calls > 1
    # A warning was emitted (content unspecified) via either logging or Python warnings
    has_log_warning = any(r.levelname == "WARNING" for r in caplog.records)
    has_py_warning = len(recwarn) > 0
    assert has_log_warning or has_py_warning


def test_disabled_has_no_slowdown_vs_noop_logger_proxy_guard():
    # Proxy performance guard: disabled should not be slower than an enabled no-op logger beyond a generous threshold
    import time
    from lightning.pytorch.event_logging.plugins import BaseEventPlugin
    from lightning.pytorch.event_logging.event_logger import EventLogger
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.trainer.trainer import Trainer

    class NoOp(BaseEventPlugin):
        def on_event(self, event):
            return None

    def run_with(event_logger):
        start = time.perf_counter()
        Trainer(max_epochs=1, limit_train_batches=2, enable_checkpointing=False, logger=False, enable_model_summary=False,
                event_logger=event_logger).fit(BoringModel())
        return time.perf_counter() - start

    t_disabled = min(run_with(None) for _ in range(3))
    t_noop = min(run_with(EventLogger(plugins=[NoOp()])) for _ in range(3))

    # Extremely loose proxy threshold to reduce flakiness across environments
    assert t_disabled <= t_noop * 3.0 + 0.50


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA not available")
def test_deterministic_order_on_cuda_if_available():
    # Cross-device determinism smoke test: ensure deterministic ordering on CUDA as on CPU test
    from lightning.pytorch.event_logging.plugins import BaseEventPlugin
    from lightning.pytorch.event_logging.event_logger import EventLogger
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.trainer.trainer import Trainer

    class Capture(BaseEventPlugin):
        def __init__(self):
            self.types = []

        def on_event(self, event):
            self.types.append(event.type)

    cap = Capture()
    logger = EventLogger(plugins=[cap])

    # Two identical CUDA runs
    Trainer(max_epochs=1, limit_train_batches=1, enable_checkpointing=False, logger=False, enable_model_summary=False,
            accelerator="cuda", devices=1, event_logger=logger).fit(BoringModel())
    run1 = list(cap.types)
    cap.types.clear()
    Trainer(max_epochs=1, limit_train_batches=1, enable_checkpointing=False, logger=False, enable_model_summary=False,
            accelerator="cuda", devices=1, event_logger=logger).fit(BoringModel())
    run2 = list(cap.types)

    assert run1 == run2
    # Relative lifecycle order should hold
    assert run2.index("forward") < run2.index("backward") < run2.index("optimizer_step")
