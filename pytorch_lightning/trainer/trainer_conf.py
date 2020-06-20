from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from hydra.types import ObjectConf
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig

cs = ConfigStore.instance()


@dataclass
class LightningTrainerConf:
    default_root_dir: Optional[str] = None
    gradient_clip_val: float = 0
    process_position: int = 0
    num_nodes: int = 1
    num_processes: int = 1
    gpus: Optional[Any] = None
    auto_select_gpus: bool = False
    tpu_cores: Optional[Any] = None
    log_gpu_memory: Optional[str] = None
    progress_bar_refresh_rate: int = 1
    overfit_batches: float = 0.0
    track_grad_norm: Any = -1
    check_val_every_n_epoch: int = 1
    fast_dev_run: bool = False
    accumulate_grad_batches: Any = 1
    max_epochs: int = 1000
    min_epochs: int = 1
    max_steps: Optional[int] = None
    min_steps: Optional[int] = None
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    limit_test_batches: float = 1.0
    val_check_interval: float = 1.0
    log_save_interval: int = 100
    row_log_interval: int = 50
    distributed_backend: Optional[str] = None
    precision: int = 32
    print_nan_grads: bool = False
    weights_summary: Optional[str] = "top"
    weights_save_path: Optional[str] = None
    num_sanity_val_steps: int = 2
    truncated_bptt_steps: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None
    benchmark: bool = False
    deterministic: bool = False
    reload_dataloaders_every_epoch: bool = False
    auto_lr_find: Any = False
    replace_sampler_ddp: bool = True
    terminate_on_nan: bool = False
    auto_scale_batch_size: Any = False
    prepare_data_per_node: bool = True
    amp_level: str = "O1"
    num_tpu_cores: Optional[int] = None


cs.store(group="trainer", name="trainer", node=LightningTrainerConf)


@dataclass
class ModelCheckpointConf:
    filepath: Optional[str] = None
    monitor: str = "val_loss"
    verbose: bool = False
    save_last: bool = False
    save_top_k: int = 1
    save_weights_only: bool = False
    mode: str = "auto"
    period: int = 1
    prefix: str = ""


cs.store(
    group="checkpoint",
    name="modelckpt",
    node=ObjectConf(target="pytorch_lightning.callbacks.ModelCheckpoint", params=ModelCheckpointConf()),
)


@dataclass
class EarlyStoppingConf:
    monitor: str = "val_loss"
    verbose: bool = False
    mode: str = "auto"
    patience: int = 3
    strict: bool = True
    min_delta: float = 0.0


cs.store(
    group="early_stopping",
    name="earlystop",
    node=ObjectConf(target="pytorch_lightning.callbacks.EarlyStopping", params=EarlyStoppingConf()),
)


@dataclass
class SimpleProfilerConf:
    output_filename: Optional[str] = None


@dataclass
class AdvancedProfilerConf:
    output_filename: Optional[str] = None
    line_count_restriction: float = 1.0


cs.store(
    group="profiler",
    name="simple",
    node=ObjectConf(target="pytorch_lightning.profiler.SimpleProfiler", params=SimpleProfilerConf()),
)

cs.store(
    group="profiler",
    name="advanced",
    node=ObjectConf(target="pytorch_lightning.profiler.AdvancedProfiler", params=AdvancedProfilerConf()),
)


@dataclass
class CometLoggerConf:
    api_key: Optional[str] = None
    save_dir: Optional[str] = None
    workspace: Optional[str] = None
    project_name: Optional[str] = None
    rest_api_key: Optional[str] = None
    experiment_name: Optional[str] = None
    experiment_key: Optional[str] = None


cs.store(
    group="logger",
    name="comet",
    node=ObjectConf(target="pytorch_lightning.loggers.comet.CometLogger", params=CometLoggerConf()),
)


@dataclass
class MLFlowLoggerConf:
    experiment_name: str = "default"
    tracking_uri: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None
    save_dir: Optional[str] = None


cs.store(
    group="logger",
    name="mlflow",
    node=ObjectConf(target="pytorch_lightning.loggers.mlflow.MLFlowLogger", params=MLFlowLoggerConf()),
)


@dataclass
class NeptuneLoggerConf:
    api_key: Optional[str] = None
    project_name: Optional[str] = None
    close_after_fit: Optional[bool] = True
    offline_mode: bool = False
    experiment_name: Optional[str] = None
    upload_source_files: Optional[List[str]] = None
    params: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


cs.store(
    group="logger",
    name="neptune",
    node=ObjectConf(target="pytorch_lightning.loggers.neptune.NeptuneLogger", params=NeptuneLoggerConf()),
)


@dataclass
class TensorboardLoggerConf:
    save_dir: str = ""
    name: Optional[str] = "default"
    version: Any = None


cs.store(
    group="logger",
    name="tensorboard",
    node=ObjectConf(target="pytorch_lightning.loggers.tensorboard.TensorBoardLogger", params=TensorboardLoggerConf()),
)


@dataclass
class TestTubeLoggerConf:
    save_dir: str = ""
    name: str = "default"
    description: Optional[str] = None
    debug: bool = False
    version: Optional[int] = None
    create_git_tag: bool = False


cs.store(
    group="logger",
    name="testtube",
    node=ObjectConf(target="pytorch_lightning.loggers.test_tube.TestTubeLogger", params=TestTubeLoggerConf()),
)


@dataclass
class WandbConf:
    name: Optional[str] = None
    save_dir: Optional[str] = None
    offline: bool = False
    id: Optional[str] = None
    anonymous: bool = False
    version: Optional[str] = None
    project: Optional[str] = None
    tags: Optional[List[str]] = None
    log_model: bool = False
    experiment = None
    entity = None
    group: Optional[str] = None


cs.store(
    group="logger",
    name="wandb",
    node=ObjectConf(target="pytorch_lightning.loggers.wandb.WandbLogger", params=WandbConf()),
)


@dataclass
class PLConfig(DictConfig):
    logger: Optional[ObjectConf] = None
    profiler: Optional[ObjectConf] = None
    checkpoint: Optional[ObjectConf] = None
    early_stopping: Optional[ObjectConf] = None
    trainer: LightningTrainerConf = MISSING

