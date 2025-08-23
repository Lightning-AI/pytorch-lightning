# Copyright The Lightning AI team.
# This test file provides CPU-only coverage for DeepSpeed lazy-import paths by mocking a minimal
# `deepspeed` module. It does not require GPUs or the real DeepSpeed package.

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from lightning.fabric.strategies import DeepSpeedStrategy


class _FakeLogger:
    def __init__(self):
        self.levels = []

    def setLevel(self, lvl):
        self.levels.append(lvl)


class _FakeZeroInit:
    def __init__(self, *args, **kwargs):
        # record for assertions
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def fake_deepspeed(monkeypatch):
    """Inject a minimal fake `deepspeed` package into sys.modules."""
    ds = ModuleType("deepspeed")
    # Mark as a package with a spec and path so importlib won't complain
    import importlib.machinery

    ds.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=Mock(), is_package=True)
    ds.__path__ = []  # type: ignore[attr-defined]

    # utils.logging.logger
    utils_mod = ModuleType("deepspeed.utils")
    logging_mod = ModuleType("deepspeed.utils.logging")
    utils_mod.__spec__ = importlib.machinery.ModuleSpec("deepspeed.utils", loader=Mock(), is_package=True)
    logging_mod.__spec__ = importlib.machinery.ModuleSpec("deepspeed.utils.logging", loader=Mock(), is_package=False)
    logger = _FakeLogger()
    logging_mod.logger = logger
    utils_mod.logging = logging_mod
    ds.utils = utils_mod

    # zero.Init
    zero_mod = ModuleType("deepspeed.zero")
    zero_mod.__spec__ = importlib.machinery.ModuleSpec("deepspeed.zero", loader=Mock(), is_package=False)
    zero_mod.Init = _FakeZeroInit
    ds.zero = zero_mod

    # checkpointing.configure
    checkpointing_mod = ModuleType("deepspeed.checkpointing")
    checkpointing_mod.__spec__ = importlib.machinery.ModuleSpec(
        "deepspeed.checkpointing", loader=Mock(), is_package=False
    )
    recorded = {"configure_calls": []}

    def _configure(**kwargs):
        recorded["configure_calls"].append(kwargs)

    checkpointing_mod.configure = _configure
    ds.checkpointing = checkpointing_mod

    # initialize
    recorded["initialize_calls"] = []

    def _initialize(**kwargs):
        recorded["initialize_calls"].append(kwargs)
        # return values: (engine, optimizer, _, scheduler)
        return Mock(name="engine"), Mock(name="optimizer"), None, Mock(name="scheduler")

    ds.initialize = _initialize

    # init_distributed
    recorded["init_distributed_calls"] = []

    def _init_distributed(*args, **kwargs):
        recorded["init_distributed_calls"].append((args, kwargs))

    ds.init_distributed = _init_distributed

    # install into sys.modules
    monkeypatch.setitem(sys.modules, "deepspeed", ds)
    monkeypatch.setitem(sys.modules, "deepspeed.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "deepspeed.utils.logging", logging_mod)
    monkeypatch.setitem(sys.modules, "deepspeed.zero", zero_mod)
    monkeypatch.setitem(sys.modules, "deepspeed.checkpointing", checkpointing_mod)

    # Pretend deepspeed is installed by forcing availability flag to True
    monkeypatch.setattr("lightning.fabric.strategies.deepspeed._DEEPSPEED_AVAILABLE", True, raising=False)

    return ds, logger, recorded


def _make_strategy_with_defaults():
    # Use defaults; we'll tweak attributes per test as needed
    return DeepSpeedStrategy()


def _get_backend() -> str:
    # simple helper used to override strategy._get_process_group_backend
    return "gloo"


def test_module_sharded_context_sets_logger_and_returns_zero_init(fake_deepspeed):
    ds_mod, logger, recorded = fake_deepspeed

    strategy = _make_strategy_with_defaults()
    # The context asserts that the config was initialized
    strategy._config_initialized = True  # type: ignore[attr-defined]

    ctx = strategy.module_sharded_context()
    assert isinstance(ctx, _FakeZeroInit)
    # logger.setLevel should be called at least once
    assert len(logger.levels) >= 1


def test_initialize_engine_import_and_logger_and_call(fake_deepspeed):
    ds_mod, logger, recorded = fake_deepspeed

    strategy = _make_strategy_with_defaults()
    # root_device.index is read; use a CUDA device number even on CPU-only hosts (no allocation happens)
    import torch

    strategy.parallel_devices = [torch.device("cuda", 0)]  # type: ignore[attr-defined]

    class _Param:
        requires_grad = True

    model = Mock()
    model.parameters.return_value = [_Param()]

    engine, optimizer, scheduler = strategy._initialize_engine(model)

    # assertions
    assert len(logger.levels) >= 1
    assert recorded["initialize_calls"], "deepspeed.initialize was not called"
    call = recorded["initialize_calls"][0]
    assert call["config"] == strategy.config
    assert call["model"] is model
    assert call["dist_init_required"] is False
    # returned mocks are propagated
    from unittest.mock import Mock as _M

    assert isinstance(engine, _M)
    assert engine._mock_name == "engine"
    assert isinstance(optimizer, _M)
    assert optimizer._mock_name == "optimizer"
    assert isinstance(scheduler, _M)
    assert scheduler._mock_name == "scheduler"


def test_init_deepspeed_distributed_calls_import_and_init(fake_deepspeed, monkeypatch):
    ds_mod, logger, recorded = fake_deepspeed

    strategy = _make_strategy_with_defaults()

    # minimal cluster env
    class _CE:
        main_port = 12345
        main_address = "127.0.0.1"

        def global_rank(self):
            return 0

        def local_rank(self):
            return 0

        def node_rank(self):
            return 0

        def world_size(self):
            return 1

        def teardown(self):
            pass

    strategy.cluster_environment = _CE()
    strategy._process_group_backend = "gloo"  # avoid CUDA requirement
    strategy._timeout = 300  # type: ignore[attr-defined]

    strategy._get_process_group_backend = _get_backend  # type: ignore[assignment]

    # ensure non-Windows path
    monkeypatch.setattr("platform.system", lambda: "Linux")

    strategy._init_deepspeed_distributed()

    assert len(logger.levels) >= 1
    assert recorded["init_distributed_calls"], "deepspeed.init_distributed was not called"
    args, kwargs = recorded["init_distributed_calls"][0]
    assert args[0] == "gloo"
    assert kwargs["distributed_port"] == 12345
    assert "timeout" in kwargs


def test_set_deepspeed_activation_checkpointing_configured(fake_deepspeed):
    ds_mod, logger, recorded = fake_deepspeed

    strategy = _make_strategy_with_defaults()
    # ensure config contains activation_checkpointing keys
    assert isinstance(strategy.config, dict)
    strategy.config.setdefault("activation_checkpointing", {})
    strategy.config["activation_checkpointing"].update({
        "partition_activations": True,
        "contiguous_memory_optimization": False,
        "cpu_checkpointing": True,
        "profile": False,
    })

    strategy._set_deepspeed_activation_checkpointing()

    assert len(logger.levels) >= 1
    assert recorded["configure_calls"], "deepspeed.checkpointing.configure was not called"
    cfg = recorded["configure_calls"][0]
    assert cfg["partition_activations"] is True
    assert cfg["contiguous_checkpointing"] is False
    assert cfg["checkpoint_in_cpu"] is True
    assert cfg["profile"] is False
