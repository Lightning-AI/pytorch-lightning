from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore
from hydra.types import ObjectConf

cs = ConfigStore.instance()


@dataclass
class AdamConf:
    betas: tuple = (0.9, 0.999)
    lr: float = 1e-3
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False


cs.store(
    group="opt", name="adam", node=ObjectConf(target="torch.optim.Adam", params=AdamConf()),
)
cs.store(
    group="opt", name="adamw", node=ObjectConf(target="torch.optim.AdamW", params=AdamConf()),
)


@dataclass
class AdamaxConf:
    betas: tuple = (0.9, 0.999)
    lr: float = 1e-3
    eps: float = 1e-8
    weight_decay: float = 0


cs.store(
    group="opt", name="adamax", node=ObjectConf(target="torch.optim.Adamax", params=AdamaxConf()),
)


@dataclass
class ASGDConf:
    alpha: float = 0.75
    lr: float = 1e-3
    lambd: float = 1e-4
    t0: float = 1e6
    weight_decay: float = 0


cs.store(
    group="opt", name="asgd", node=ObjectConf(target="torch.optim.ASGD", params=ASGDConf()),
)


@dataclass
class LBFGSConf:
    lr: float = 1
    max_iter: int = 20
    max_eval: int = 25
    tolerance_grad: float = 1e-5
    tolerance_change: float = 1e-9
    history_size: int = 100
    line_search_fn: Optional[str] = None


cs.store(
    group="opt", name="lbfgs", node=ObjectConf(target="torch.optim.LBFGS", params=LBFGSConf()),
)


@dataclass
class RMSpropConf:
    lr: float = 1e-2
    momentum: float = 0
    alpha: float = 0.99
    eps: float = 1e-8
    centered: bool = True
    weight_decay: float = 0


cs.store(
    group="opt", name="rmsprop", node=ObjectConf(target="torch.optim.RMSprop", params=RMSpropConf()),
)


@dataclass
class RpropConf:
    lr: float = 1e-2
    etas: tuple = (0.5, 1.2)
    step_sizes: tuple = (1e-6, 50)


cs.store(
    group="opt", name="rprop", node=ObjectConf(target="torch.optim.Rprop", params=RpropConf()),
)


@dataclass
class SGDConf:
    lr: float = 1e-2
    momentum: float = 0
    weight_decay: float = 0
    dampening: float = 0
    nesterov: bool = False


cs.store(
    group="opt", name="sgd", node=ObjectConf(target="torch.optim.SGD", params=SGDConf()),
)
