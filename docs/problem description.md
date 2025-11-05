Unified Event‑Logging Framework for PyTorch Lightning

Problem
Lightning lacks a single, deterministic pipeline to capture structured training events (forward/backward, optimizer steps, metrics, checkpoints). Signals are split across callbacks, profilers, and loggers, making runs hard to compare and extend safely. Build a unified, pluggable event logger that emits consistent records and can be enabled or dropped in without changing core training code. When disabled, it must introduce no measurable overhead.

Required behaviors
- EventRecord: type (str), timestamp, duration (optional), metadata (dict).
- Lifecycle coverage and order: forward → backward → optimizer_step; also emit metric and checkpoint events. Ordering must be deterministic across runs/devices.
- Plugin dispatch: users provide an ordered list; dispatch follows exactly that list order for every event. If a plugin raises, quarantine it, emit a WARNING, continue with the rest.
- Config surface: Trainer accepts event_logger (optional). EventLogger(plugins: list|tuple = (), dry_run: bool = False). In dry_run, events are dropped and plugins are not invoked (guaranteed no side effects).
- Backward compatibility: event_logger=None preserves current behavior and performance.
- Telemetry (per-event/per-plugin timings) is optional; not required for acceptance.

Public surface and locations
- src/lightning/pytorch/event_logging/types.py: EventRecord
- src/lightning/pytorch/event_logging/plugins.py: BaseEventPlugin.on_event(self, event: EventRecord) -> None
- src/lightning/pytorch/event_logging/event_logger.py: EventLogger
- Canonical event types: "forward", "backward", "optimizer_step", "metric", "checkpoint" (within-step order as above).
- Usage: Trainer(..., event_logger=EventLogger(plugins=[MyPlugin(), OtherPlugin()], dry_run=False)).
 - Disabling: Trainer(event_logger=None) disables event logging (no plugin invocation, no overhead).

Acceptance criteria
- Deterministic event sequences (including plugin dispatch order), explicit presence of core lifecycle/metric/checkpoint events, plugin quarantine with a WARNING, correct dry_run semantics (no plugin calls), and zero-overhead when disabled.