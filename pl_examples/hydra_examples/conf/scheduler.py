from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from hydra.types import ObjectConf

cs = ConfigStore.instance()


@dataclass
class CosineConf:
    T_max: int = 100
    eta_min: float = 0
    last_epoch: int = -1


cs.store(
    group="scheduler",
    name="cosine",
    node=ObjectConf(target="torch.optim.lr_scheduler.CosineAnnealingLR", params=CosineConf()),
)


@dataclass
class CosineWarmConf:
    T_0: int = 10
    T_mult: int = 1
    eta_min: float = 0
    last_epoch: int = -1


cs.store(
    group="scheduler",
    name="cosinewarm",
    node=ObjectConf(target="torch.optim.lr_scheduler.CosineAnnealingLR", params=CosineWarmConf()),
)


@dataclass
class CyclicConf:
    base_lr: Any = 1e-3
    max_lr: Any = 1e-2
    step_size_up: int = 2000
    step_size_down: int = 2000
    mode: str = "triangular"
    gamma: float = 1
    scale_fn: Optional[Any] = None
    scal_mode: str = "cycle"
    cycle_momentum: bool = True
    base_momentum: Any = 0.8
    max_momentum: Any = 0.9
    last_epoch: int = -1


cs.store(
    group="scheduler", name="cyclic", node=ObjectConf(target="torch.optim.lr_scheduler.CyclicLR", params=CyclicConf()),
)


@dataclass
class ExponentialConf:
    gamma: float = 1
    last_epoch: int = -1


cs.store(
    group="scheduler",
    name="exponential",
    node=ObjectConf(target="torch.optim.lr_scheduler.ExponentialLR", params=ExponentialConf()),
)


@dataclass
class RedPlatConf:
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    verbose: bool = False
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: Any = 0
    eps: float = 1e-8


cs.store(
    group="scheduler",
    name="redplat",
    node=ObjectConf(target="torch.optim.lr_scheduler.ReduceLROnPlateau", params=RedPlatConf()),
)


@dataclass
class MultiStepConf:
    milestones: List = field(default_factory=lambda: [10, 20, 30, 40])
    gamma: float = 0.1
    last_epoch: int = -1


cs.store(
    group="scheduler",
    name="multistep",
    node=ObjectConf(target="torch.optim.lr_scheduler.MultiStepLR", params=MultiStepConf()),
)


@dataclass
class OneCycleConf:
    max_lr: Any = 1e-2
    total_steps: int = 2000
    epochs: int = 200
    steps_per_epoch: int = 100
    pct_start: float = 0.3
    anneal_strategy: str = "cos"
    cycle_momentum: bool = True
    base_momentum: Any = 0.8
    max_momentum: Any = 0.9
    div_factor: float = 25
    final_div_factor: float = 1e4
    last_epoch: int = -1


cs.store(
    group="scheduler",
    name="onecycle",
    node=ObjectConf(target="torch.optim.lr_scheduler.OneCycleLR", params=OneCycleConf()),
)


@dataclass
class StepConf:
    step_size: int = 20
    gamma: float = 0.1
    last_epoch: int = -1


cs.store(
    group="scheduler", name="step", node=ObjectConf(target="torch.optim.lr_scheduler.StepLR", params=StepConf()),
)
