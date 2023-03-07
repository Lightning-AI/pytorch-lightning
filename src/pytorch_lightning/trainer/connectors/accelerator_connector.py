# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from collections import Counter
from typing import cast, Dict, List, Optional, Union

import torch
from typing_extensions import get_args, Literal

from lightning_fabric.plugins.environments import (
    ClusterEnvironment,
    KubeflowEnvironment,
    LightningEnvironment,
    LSFEnvironment,
    SLURMEnvironment,
    TorchElasticEnvironment,
)
from lightning_fabric.utilities.device_parser import _determine_root_gpu_device
from lightning_fabric.utilities.imports import _IS_INTERACTIVE, _TORCH_GREATER_EQUAL_1_11
from pytorch_lightning.accelerators import AcceleratorRegistry
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.accelerators.cpu import CPUAccelerator
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.accelerators.hpu import HPUAccelerator
from pytorch_lightning.accelerators.ipu import _IPU_AVAILABLE, IPUAccelerator
from pytorch_lightning.accelerators.mps import MPSAccelerator
from pytorch_lightning.accelerators.tpu import TPUAccelerator
from pytorch_lightning.plugins import (
    ApexMixedPrecisionPlugin,
    CheckpointIO,
    ColossalAIPrecisionPlugin,
    DeepSpeedPrecisionPlugin,
    DoublePrecisionPlugin,
    FullyShardedNativeMixedPrecisionPlugin,
    HPUPrecisionPlugin,
    IPUPrecisionPlugin,
    MixedPrecisionPlugin,
    PLUGIN_INPUT,
    PrecisionPlugin,
    ShardedNativeMixedPrecisionPlugin,
    TPUBf16PrecisionPlugin,
    TPUPrecisionPlugin,
)
from pytorch_lightning.plugins.environments import BaguaEnvironment
from pytorch_lightning.plugins.layer_sync import LayerSync, NativeSyncBatchNorm
from pytorch_lightning.plugins.precision.fsdp_native_native_amp import FullyShardedNativeNativeMixedPrecisionPlugin
from pytorch_lightning.strategies import (
    ColossalAIStrategy,
    DDPFullyShardedNativeStrategy,
    DDPFullyShardedStrategy,
    DDPShardedStrategy,
    DDPSpawnShardedStrategy,
    DDPSpawnStrategy,
    DDPStrategy,
    DeepSpeedStrategy,
    HorovodStrategy,
    HPUParallelStrategy,
    IPUStrategy,
    ParallelStrategy,
    SingleDeviceStrategy,
    SingleHPUStrategy,
    SingleTPUStrategy,
    Strategy,
    StrategyRegistry,
    TPUSpawnStrategy,
)
from pytorch_lightning.strategies.ddp_spawn import _DDP_FORK_ALIASES
from pytorch_lightning.strategies.horovod import _HOROVOD_AVAILABLE
from pytorch_lightning.tuner.auto_gpu_select import pick_multiple_gpus
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_info, rank_zero_warn

log = logging.getLogger(__name__)

if _HOROVOD_AVAILABLE:
    import horovod.torch as hvd

_LITERAL_WARN = Literal["warn"]
_PRECISION_INPUT_INT = Literal[64, 32, 16]
_PRECISION_INPUT_STR = Literal["64", "32", "16", "bf16"]
_PRECISION_INPUT = Union[_PRECISION_INPUT_INT, _PRECISION_INPUT_STR]


class AcceleratorConnector:
    def __init__(
        self,
        devices: Optional[Union[List[int], str, int]] = None,
        num_nodes: int = 1,
        accelerator: Optional[Union[str, Accelerator]] = None,
        strategy: Optional[Union[str, Strategy]] = None,
        plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = None,
        precision: _PRECISION_INPUT = 32,
        amp_type: Optional[str] = None,
        amp_level: Optional[str] = None,
        sync_batchnorm: bool = False,
        benchmark: Optional[bool] = None,
        replace_sampler_ddp: bool = True,
        deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
        auto_select_gpus: Optional[bool] = None,  # TODO: Remove in v2.0.0
        num_processes: Optional[int] = None,  # TODO: Remove in v2.0.0
        tpu_cores: Optional[Union[List[int], str, int]] = None,  # TODO: Remove in v2.0.0
        ipus: Optional[int] = None,  # TODO: Remove in v2.0.0
        gpus: Optional[Union[List[int], str, int]] = None,  # TODO: Remove in v2.0.0
    ) -> None:
        """The AcceleratorConnector parses several Trainer arguments and instantiates the Strategy including other
        components such as the Accelerator and Precision plugins.

            A. accelerator flag could be:
                1. accelerator class
                2. accelerator str
                3. accelerator auto

            B. strategy flag could be :
                1. strategy class
                2. strategy str registered with StrategyRegistry
                3. strategy str in _strategy_type enum which listed in each strategy as
                   backend (registed these too, and _strategy_type could be deprecated)

            C. plugins flag could be:
                1. List of str, which could contain:
                    i. precision str (Not supported in the old accelerator_connector version)
                    ii. checkpoint_io str (Not supported in the old accelerator_connector version)
                    iii. cluster_environment str (Not supported in the old accelerator_connector version)
                2. List of class, which could contains:
                    i. precision class (should be removed, and precision flag should allow user pass classes)
                    ii. checkpoint_io class
                    iii. cluster_environment class


        priorities which to take when:
            A. Class > str
            B. Strategy > Accelerator/precision/plugins
        """
        if deterministic:
            if benchmark is None:
                # Set benchmark to False to ensure determinism
                benchmark = False
            elif benchmark:
                rank_zero_warn(
                    "You passed `deterministic=True` and `benchmark=True`. Note that PyTorch ignores"
                    " torch.backends.cudnn.deterministic=True when torch.backends.cudnn.benchmark=True.",
                )
        # TODO: move to gpu accelerator
        if benchmark is not None:
            torch.backends.cudnn.benchmark = benchmark
        self.benchmark = torch.backends.cudnn.benchmark
        self.replace_sampler_ddp = replace_sampler_ddp
        self._init_deterministic(deterministic)

        # 1. Parsing flags
        # Get registered strategies, built-in accelerators and precision plugins
        self._registered_strategies = StrategyRegistry.available_strategies()
        self._accelerator_types = AcceleratorRegistry.available_accelerators()

        # Raise an exception if there are conflicts between flags
        # Set each valid flag to `self._x_flag` after validation
        # For devices: Assign gpus, ipus, etc. to the accelerator flag and devices flag
        self._strategy_flag: Optional[Union[Strategy, str]] = None
        self._accelerator_flag: Optional[Union[Accelerator, str]] = None
        self._precision_flag: _PRECISION_INPUT_STR = "32"
        self._precision_plugin_flag: Optional[PrecisionPlugin] = None
        self._cluster_environment_flag: Optional[Union[ClusterEnvironment, str]] = None
        self._parallel_devices: List[Union[int, torch.device, str]] = []
        self._layer_sync: Optional[LayerSync] = NativeSyncBatchNorm() if sync_batchnorm else None
        self.checkpoint_io: Optional[CheckpointIO] = None
        self._amp_type_flag: Optional[str] = None  # TODO: Remove in v2.0.0
        self._amp_level_flag: Optional[str] = amp_level  # TODO: Remove in v2.0.0
        self._auto_select_gpus: Optional[bool] = auto_select_gpus  # TODO: Remove in v2.0.0

        self._check_config_and_set_final_flags(
            strategy=strategy,
            accelerator=accelerator,
            precision=precision,
            plugins=plugins,
            amp_type=amp_type,
            amp_level=amp_level,
            sync_batchnorm=sync_batchnorm,
        )
        self._check_device_config_and_set_final_flags(
            devices=devices, num_nodes=num_nodes, num_processes=num_processes, gpus=gpus, ipus=ipus, tpu_cores=tpu_cores
        )
        # 2. Instantiate Accelerator
        self._set_accelerator_if_ipu_strategy_is_passed()

        # handle `auto`, `None` and `gpu`
        if self._accelerator_flag == "auto" or self._accelerator_flag is None:
            self._accelerator_flag = self._choose_auto_accelerator()
        elif self._accelerator_flag == "gpu":
            self._accelerator_flag = self._choose_gpu_accelerator_backend()

        self._set_parallel_devices_and_init_accelerator()

        # 3. Instantiate ClusterEnvironment
        self.cluster_environment: ClusterEnvironment = self._choose_and_init_cluster_environment()

        # 4. Instantiate Strategy - Part 1
        if self._strategy_flag in (None, "auto"):
            self._strategy_flag = self._choose_strategy()
        # In specific cases, ignore user selection and fall back to a different strategy
        self._check_strategy_and_fallback()
        self._init_strategy()

        # 5. Instantiate Precision Plugin
        self.precision_plugin = self._check_and_init_precision()

        # 6. Instantiate Strategy - Part 2
        self._lazy_init_strategy()

    def _init_deterministic(self, deterministic: Optional[Union[bool, _LITERAL_WARN]]) -> None:
        self.deterministic = deterministic or False  # default to False if not set
        if _TORCH_GREATER_EQUAL_1_11 and deterministic == "warn":
            torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            torch.use_deterministic_algorithms(self.deterministic)
        if self.deterministic:
            # fixing non-deterministic part of horovod
            # https://github.com/Lightning-AI/lightning/pull/1572/files#r420279383
            os.environ["HOROVOD_FUSION_THRESHOLD"] = "0"

            # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def _check_config_and_set_final_flags(
        self,
        strategy: Optional[Union[str, Strategy]],
        accelerator: Optional[Union[str, Accelerator]],
        precision: _PRECISION_INPUT,
        plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]],
        amp_type: Optional[str],
        amp_level: Optional[str],
        sync_batchnorm: bool,
    ) -> None:
        """This method checks:

        1. strategy: whether the strategy name is valid, and sets the internal flags if it is.
        2. accelerator: if the value of the accelerator argument is a type of accelerator (instance or string),
            set self._accelerator_flag accordingly.
        3. precision: The final value of the precision flag may be determined either by the precision argument or
            by a plugin instance.
        4. plugins: The list of plugins may contain a Precision plugin, CheckpointIO, ClusterEnvironment and others.
            Additionally, other flags such as `precision` or `sync_batchnorm` can populate the list with the
            corresponding plugin instances.
        """
        if plugins is not None:
            plugins = [plugins] if not isinstance(plugins, list) else plugins

        if isinstance(strategy, str):
            strategy = strategy.lower()

        if strategy is not None:
            self._strategy_flag = strategy
            if strategy == "ddp_cpu":
                raise MisconfigurationException(
                    "`Trainer(strategy='ddp_cpu')` is not a valid strategy,"
                    " you can use `Trainer(strategy='ddp'|'ddp_spawn'|'ddp_fork', accelerator='cpu')` instead."
                )
            if strategy == "tpu_spawn":
                raise MisconfigurationException(
                    "`Trainer(strategy='tpu_spawn')` is not a valid strategy,"
                    " you can use `Trainer(strategy='ddp_spawn', accelerator='tpu')` instead."
                )

        if (
            strategy not in (None, "auto")
            and strategy not in self._registered_strategies
            and not isinstance(strategy, Strategy)
        ):
            raise ValueError(
                f"You selected an invalid strategy name: `strategy={strategy!r}`."
                " It must be either a string or an instance of `pytorch_lightning.strategies.Strategy`."
                " Example choices: ddp, ddp_spawn, deepspeed, dp, ..."
                " Find a complete list of options in our documentation at https://lightning.ai"
            )

        if (
            accelerator is not None
            and accelerator not in self._accelerator_types
            and accelerator not in ("auto", "gpu")
            and not isinstance(accelerator, Accelerator)
        ):
            raise ValueError(
                f"You selected an invalid accelerator name: `accelerator={accelerator!r}`."
                f" Available names are: {', '.join(self._accelerator_types)}."
            )

        # MPS accelerator is incompatible with DDP family of strategies. It supports single-device operation only.
        is_ddp_str = isinstance(strategy, str) and "ddp" in strategy
        is_dp_str = isinstance(strategy, str) and "dp" in strategy
        is_deepspeed_str = isinstance(strategy, str) and "deepspeed" in strategy
        is_parallel_strategy = isinstance(strategy, ParallelStrategy) or is_ddp_str or is_dp_str or is_deepspeed_str
        is_mps_accelerator = MPSAccelerator.is_available() and (
            accelerator in ("mps", "auto", "gpu", None) or isinstance(accelerator, MPSAccelerator)
        )
        if is_mps_accelerator and is_parallel_strategy:
            raise ValueError(
                f"You set `strategy={strategy}` but strategies from the DDP family are not supported on the"
                f" MPS accelerator. Either explicitly set `accelerator='cpu'` or change the strategy."
            )

        self._accelerator_flag = accelerator

        supported_precision = get_args(_PRECISION_INPUT_STR) + get_args(_PRECISION_INPUT_INT)
        if precision not in supported_precision:
            raise MisconfigurationException(
                f"Precision {repr(precision)} is invalid. Allowed precision values: {supported_precision}"
            )
        self._precision_flag = cast(_PRECISION_INPUT_STR, str(precision))

        if plugins:
            plugins_flags_types: Dict[str, int] = Counter()
            for plugin in plugins:
                if isinstance(plugin, PrecisionPlugin):
                    self._precision_plugin_flag = plugin
                    plugins_flags_types[PrecisionPlugin.__name__] += 1
                elif isinstance(plugin, CheckpointIO):
                    self.checkpoint_io = plugin
                    plugins_flags_types[CheckpointIO.__name__] += 1
                elif isinstance(plugin, ClusterEnvironment):
                    self._cluster_environment_flag = plugin
                    plugins_flags_types[ClusterEnvironment.__name__] += 1
                elif isinstance(plugin, LayerSync):
                    if sync_batchnorm and not isinstance(plugin, NativeSyncBatchNorm):
                        raise MisconfigurationException(
                            f"You set `Trainer(sync_batchnorm=True)` and provided a `{plugin.__class__.__name__}`"
                            " plugin, but this is not allowed. Choose one or the other."
                        )
                    self._layer_sync = plugin
                    plugins_flags_types[NativeSyncBatchNorm.__name__] += 1
                else:
                    raise MisconfigurationException(
                        f"Found invalid type for plugin {plugin}. Expected one of: PrecisionPlugin, "
                        "CheckpointIO, ClusterEnviroment, or LayerSync."
                    )

            duplicated_plugin_key = [k for k, v in plugins_flags_types.items() if v > 1]
            if duplicated_plugin_key:
                raise MisconfigurationException(
                    f"Received multiple values for {', '.join(duplicated_plugin_key)} flags in `plugins`."
                    " Expected one value for each type at most."
                )

        # handle the case when the user passes in a strategy instance which has an accelerator, precision,
        # checkpoint io or cluster env set up
        # TODO: improve the error messages below
        if self._strategy_flag and isinstance(self._strategy_flag, Strategy):
            if self._strategy_flag._accelerator:
                if self._accelerator_flag:
                    raise MisconfigurationException(
                        "accelerator set through both strategy class and accelerator flag, choose one"
                    )
                else:
                    self._accelerator_flag = self._strategy_flag._accelerator
            if self._strategy_flag._precision_plugin:
                # [RFC] handle precision plugin set up conflict?
                if self._precision_plugin_flag:
                    raise MisconfigurationException("precision set through both strategy class and plugins, choose one")
                else:
                    self._precision_plugin_flag = self._strategy_flag._precision_plugin
            if self._strategy_flag._checkpoint_io:
                if self.checkpoint_io:
                    raise MisconfigurationException(
                        "checkpoint_io set through both strategy class and plugins, choose one"
                    )
                else:
                    self.checkpoint_io = self._strategy_flag._checkpoint_io
            if getattr(self._strategy_flag, "cluster_environment", None):
                if self._cluster_environment_flag:
                    raise MisconfigurationException(
                        "cluster_environment set through both strategy class and plugins, choose one"
                    )
                else:
                    self._cluster_environment_flag = getattr(self._strategy_flag, "cluster_environment")

            if hasattr(self._strategy_flag, "parallel_devices"):
                if self._strategy_flag.parallel_devices:
                    if self._strategy_flag.parallel_devices[0].type == "cpu":
                        if self._accelerator_flag and self._accelerator_flag not in ("auto", "cpu"):
                            raise MisconfigurationException(
                                f"CPU parallel_devices set through {self._strategy_flag.__class__.__name__} class,"
                                f" but accelerator set to {self._accelerator_flag}, please choose one device type"
                            )
                        self._accelerator_flag = "cpu"
                    if self._strategy_flag.parallel_devices[0].type == "cuda":
                        if self._accelerator_flag and self._accelerator_flag not in ("auto", "cuda", "gpu"):
                            raise MisconfigurationException(
                                f"GPU parallel_devices set through {self._strategy_flag.__class__.__name__} class,"
                                f" but accelerator set to {self._accelerator_flag}, please choose one device type"
                            )
                        self._accelerator_flag = "cuda"
                    self._parallel_devices = self._strategy_flag.parallel_devices

        if amp_type is not None:
            rank_zero_deprecation(
                "The NVIDIA/apex AMP implementation has been deprecated upstream. Consequently, its integration inside"
                " PyTorch Lightning has been deprecated in v1.9.0 and will be removed in v2.0.0."
                f" The `Trainer(amp_backend={amp_type!r})` argument is deprecated. Removing this argument will avoid"
                f" this message, it will select PyTorch's implementation automatically."
            )
        else:
            amp_type = None
        self._amp_type_flag = amp_type

        if amp_level is not None:
            rank_zero_deprecation(
                "The NVIDIA/apex AMP implementation has been deprecated upstream. Consequently, its integration inside"
                " PyTorch Lightning has been deprecated in v1.9.0 and will be removed in v2.0.0."
                f" The `Trainer(amp_level={amp_level!r})` argument is deprecated. Removing this argument will avoid"
                f" this message."
            )
            if self._amp_type_flag != "apex":
                raise MisconfigurationException(
                    f"You have asked for `amp_level={amp_level!r}` but it's only supported with `amp_backend='apex'`."
                )

    def _check_device_config_and_set_final_flags(
        self,
        devices: Optional[Union[List[int], str, int]],
        num_nodes: int,
        num_processes: Optional[int],
        gpus: Optional[Union[List[int], str, int]],
        ipus: Optional[int],
        tpu_cores: Optional[Union[List[int], str, int]],
    ) -> None:
        self._num_nodes_flag = int(num_nodes) if num_nodes is not None else 1
        self._devices_flag = devices

        if self._devices_flag in ([], 0, "0"):
            accelerator_name = (
                self._accelerator_flag.__class__.__qualname__
                if isinstance(self._accelerator_flag, Accelerator)
                else self._accelerator_flag
            )
            raise MisconfigurationException(
                f"`Trainer(devices={self._devices_flag!r})` value is not a valid input"
                f" using {accelerator_name} accelerator."
            )

        # TODO: Delete this method when num_processes, gpus, ipus and tpu_cores gets removed
        self._map_deprecated_devices_specific_info_to_accelerator_and_device_flag(
            devices, num_processes, gpus, ipus, tpu_cores
        )

        if self._devices_flag == "auto" and self._accelerator_flag is None:
            raise MisconfigurationException(
                f"You passed `devices={devices}` but haven't specified"
                " `accelerator=('auto'|'tpu'|'gpu'|'ipu'|'cpu'|'hpu'|'mps')` for the devices mapping."
            )

    def _map_deprecated_devices_specific_info_to_accelerator_and_device_flag(
        self,
        devices: Optional[Union[List[int], str, int]],
        num_processes: Optional[int],
        gpus: Optional[Union[List[int], str, int]],
        ipus: Optional[int],
        tpu_cores: Optional[Union[List[int], str, int]],
    ) -> None:
        """Emit deprecation warnings for num_processes, gpus, ipus, tpu_cores and set the `devices_flag` and
        `accelerator_flag`."""
        if num_processes is not None:
            rank_zero_deprecation(
                f"Setting `Trainer(num_processes={num_processes})` is deprecated in v1.7 and will be removed"
                f" in v2.0. Please use `Trainer(accelerator='cpu', devices={num_processes})` instead."
            )
        if gpus is not None:
            rank_zero_deprecation(
                f"Setting `Trainer(gpus={gpus!r})` is deprecated in v1.7 and will be removed"
                f" in v2.0. Please use `Trainer(accelerator='gpu', devices={gpus!r})` instead."
            )
        if tpu_cores is not None:
            rank_zero_deprecation(
                f"Setting `Trainer(tpu_cores={tpu_cores!r})` is deprecated in v1.7 and will be removed"
                f" in v2.0. Please use `Trainer(accelerator='tpu', devices={tpu_cores!r})` instead."
            )
        if ipus is not None:
            rank_zero_deprecation(
                f"Setting `Trainer(ipus={ipus})` is deprecated in v1.7 and will be removed"
                f" in v2.0. Please use `Trainer(accelerator='ipu', devices={ipus})` instead."
            )
        self._gpus: Optional[Union[List[int], str, int]] = gpus
        self._tpu_cores: Optional[Union[List[int], str, int]] = tpu_cores
        deprecated_devices_specific_flag = num_processes or gpus or ipus or tpu_cores
        if deprecated_devices_specific_flag and deprecated_devices_specific_flag not in ([], 0, "0"):
            if devices:
                # TODO improve error message
                rank_zero_warn(
                    f"The flag `devices={devices}` will be ignored, "
                    f"instead the device specific number {deprecated_devices_specific_flag} will be used"
                )

            if [(num_processes is not None), (gpus is not None), (ipus is not None), (tpu_cores is not None)].count(
                True
            ) > 1:
                # TODO: improve error message
                rank_zero_warn("more than one device specific flag has been set")
            self._devices_flag = deprecated_devices_specific_flag

            if self._accelerator_flag is None:
                # set accelerator type based on num_processes, gpus, ipus, tpu_cores
                if ipus:
                    self._accelerator_flag = "ipu"
                if tpu_cores:
                    self._accelerator_flag = "tpu"
                if gpus:
                    self._accelerator_flag = "cuda"
                if num_processes:
                    self._accelerator_flag = "cpu"

    def _set_accelerator_if_ipu_strategy_is_passed(self) -> None:
        # current logic only apply to object config
        # TODO this logic should apply to both str and object config
        if isinstance(self._strategy_flag, IPUStrategy):
            self._accelerator_flag = "ipu"

    def _choose_auto_accelerator(self) -> str:
        """Choose the accelerator type (str) based on availability when ``accelerator='auto'``."""
        if self._accelerator_flag == "auto":
            if TPUAccelerator.is_available():
                return "tpu"
            if _IPU_AVAILABLE:
                return "ipu"
            if HPUAccelerator.is_available():
                return "hpu"
            if MPSAccelerator.is_available():
                return "mps"
            if CUDAAccelerator.is_available():
                return "cuda"
        return "cpu"

    @staticmethod
    def _choose_gpu_accelerator_backend() -> str:
        if MPSAccelerator.is_available():
            return "mps"
        if CUDAAccelerator.is_available():
            return "cuda"

        raise MisconfigurationException("No supported gpu backend found!")

    def _set_parallel_devices_and_init_accelerator(self) -> None:
        if isinstance(self._accelerator_flag, Accelerator):
            self.accelerator: Accelerator = self._accelerator_flag
        else:
            assert self._accelerator_flag is not None
            self.accelerator = AcceleratorRegistry.get(self._accelerator_flag)
        accelerator_cls = self.accelerator.__class__

        if not accelerator_cls.is_available():
            available_accelerator = [
                acc_str
                for acc_str in self._accelerator_types
                if AcceleratorRegistry[acc_str]["accelerator"].is_available()
            ]
            raise MisconfigurationException(
                f"`{accelerator_cls.__qualname__}` can not run on your system"
                " since the accelerator is not available. The following accelerator(s)"
                " is available and can be passed into `accelerator` argument of"
                f" `Trainer`: {available_accelerator}."
            )

        self._set_devices_flag_if_auto_passed()

        self._gpus = self._devices_flag if not self._gpus else self._gpus
        self._tpu_cores = self._devices_flag if not self._tpu_cores else self._tpu_cores

        self._set_devices_flag_if_auto_select_gpus_passed()

        self._devices_flag = accelerator_cls.parse_devices(self._devices_flag)
        if not self._parallel_devices:
            self._parallel_devices = accelerator_cls.get_parallel_devices(self._devices_flag)

    def _set_devices_flag_if_auto_passed(self) -> None:
        if self._devices_flag == "auto" or self._devices_flag is None:
            self._devices_flag = self.accelerator.auto_device_count()

    def _set_devices_flag_if_auto_select_gpus_passed(self) -> None:
        if self._auto_select_gpus is not None:
            rank_zero_deprecation(
                "The Trainer argument `auto_select_gpus` has been deprecated in v1.9.0 and will be removed in v2.0.0."
                " Please use the function `pytorch_lightning.accelerators.find_usable_cuda_devices` instead."
            )
        if self._auto_select_gpus and isinstance(self._gpus, int) and isinstance(self.accelerator, CUDAAccelerator):
            self._devices_flag = pick_multiple_gpus(
                self._gpus,
                # we already show a deprecation message when user sets Trainer(auto_select_gpus=...)
                _show_deprecation=False,
            )
            log.info(f"Auto select gpus: {self._devices_flag}")

    def _choose_and_init_cluster_environment(self) -> ClusterEnvironment:
        if isinstance(self._cluster_environment_flag, ClusterEnvironment):
            return self._cluster_environment_flag
        for env_type in (
            SLURMEnvironment,
            BaguaEnvironment,
            TorchElasticEnvironment,
            KubeflowEnvironment,
            LSFEnvironment,
        ):
            if env_type.detect():
                return env_type()
        return LightningEnvironment()

    def _choose_strategy(self) -> Union[Strategy, str]:
        if self._accelerator_flag == "ipu":
            return IPUStrategy.strategy_name
        if self._accelerator_flag == "hpu":
            if self._parallel_devices and len(self._parallel_devices) > 1:
                return HPUParallelStrategy.strategy_name
            else:
                return SingleHPUStrategy(device=torch.device("hpu"))
        if self._accelerator_flag == "tpu":
            if self._parallel_devices and len(self._parallel_devices) > 1:
                return TPUSpawnStrategy.strategy_name
            else:
                # TODO: lazy initialized device, then here could be self._strategy_flag = "single_tpu_device"
                return SingleTPUStrategy(device=self._parallel_devices[0])  # type: ignore
        if _HOROVOD_AVAILABLE and ("OMPI_COMM_WORLD_RANK" in os.environ or "HOROVOD_RANK" in os.environ):
            return HorovodStrategy.strategy_name
        if self._num_nodes_flag > 1:
            return DDPStrategy.strategy_name
        if len(self._parallel_devices) <= 1:
            # TODO: Change this once gpu accelerator was renamed to cuda accelerator
            if isinstance(self._accelerator_flag, (CUDAAccelerator, MPSAccelerator)) or (
                isinstance(self._accelerator_flag, str) and self._accelerator_flag in ("cuda", "gpu", "mps")
            ):
                device = _determine_root_gpu_device(self._parallel_devices)
            else:
                device = "cpu"
            # TODO: lazy initialized device, then here could be self._strategy_flag = "single_device"
            return SingleDeviceStrategy(device=device)  # type: ignore
        if len(self._parallel_devices) > 1:
            if _IS_INTERACTIVE:
                return "ddp_fork"
            if self._strategy_flag == "auto":
                # None chooses "ddp_spawn" for backwards compatibility, auto chooses "ddp" for future compatibility
                return "ddp"
            return "ddp_spawn"

        return DDPStrategy.strategy_name

    def _check_strategy_and_fallback(self) -> None:
        """Checks edge cases when the strategy selection was a string input, and we need to fall back to a
        different choice depending on other parameters or the environment."""
        # current fallback and check logic only apply to user pass in str config and object config
        # TODO this logic should apply to both str and object config
        strategy_flag = "" if isinstance(self._strategy_flag, Strategy) else self._strategy_flag

        if strategy_flag in ("ddp_spawn", "ddp_spawn_find_unused_parameters_false") and (
            TorchElasticEnvironment.detect()
            or KubeflowEnvironment.detect()
            or SLURMEnvironment.detect()
            or LSFEnvironment.detect()
        ):
            strategy_flag = "ddp"
        if strategy_flag == "dp" and self._accelerator_flag == "cpu":
            rank_zero_warn(f"{strategy_flag!r} is not supported on CPUs, hence setting `strategy='ddp'`.")
            strategy_flag = "ddp"
        if (
            strategy_flag in DDPFullyShardedNativeStrategy.get_registered_strategies()
            or isinstance(self._strategy_flag, DDPFullyShardedNativeStrategy)
        ) and self._accelerator_flag not in ("cuda", "gpu"):
            raise MisconfigurationException(
                f"You selected strategy to be `{DDPFullyShardedNativeStrategy.strategy_name}`, "
                "but GPU accelerator is not used."
            )
        if strategy_flag in _DDP_FORK_ALIASES and "fork" not in torch.multiprocessing.get_all_start_methods():
            raise ValueError(
                f"You selected `Trainer(strategy='{strategy_flag}')` but process forking is not supported on this"
                f" platform. We recommed `Trainer(strategy='ddp_spawn')` instead."
            )
        if strategy_flag:
            self._strategy_flag = strategy_flag

    def _handle_horovod(self) -> None:
        if self._num_nodes_flag > 1:
            raise MisconfigurationException(
                "Horovod does not support setting num_nodes / num_gpus explicitly. Use "
                "horovodrun / mpirun to configure the number of processes."
            )

        if not _HOROVOD_AVAILABLE:
            raise MisconfigurationException(
                'Requested `strategy="horovod"`, but Horovod is not installed.'
                " Install with `HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]`"
            )

        hvd.init()
        if isinstance(self.accelerator, CUDAAccelerator):
            # Horovod assigns one local GPU per process
            self._parallel_devices = [torch.device(f"cuda:{i}") for i in range(hvd.local_size())]
        else:
            self._parallel_devices = [torch.device("cpu")] * hvd.local_size()

    def _init_strategy(self) -> None:
        """Instantiate the Strategy given depending on the setting of ``_strategy_flag``."""
        if isinstance(self._strategy_flag, HorovodStrategy) or self._strategy_flag == "horovod":
            # handle horovod has to happen before initialize strategy because HorovodStrategy needs hvd.init() first.
            # TODO lazy initialized and setup horovod strategy `global_rank`
            self._handle_horovod()
        # The validation of `_strategy_flag` already happened earlier on in the connector
        assert isinstance(self._strategy_flag, (str, Strategy))
        if isinstance(self._strategy_flag, str):
            self.strategy = StrategyRegistry.get(self._strategy_flag)
        else:
            # TODO(fabric): remove ignore after merging Fabric and PL strategies
            self.strategy = self._strategy_flag  # type: ignore[assignment]

    def _check_and_init_precision(self) -> PrecisionPlugin:
        self._validate_precision_choice()
        if isinstance(self._precision_plugin_flag, PrecisionPlugin):
            return self._precision_plugin_flag

        if isinstance(self.accelerator, IPUAccelerator):
            return IPUPrecisionPlugin(self._precision_flag)  # type: ignore
        if isinstance(self.accelerator, HPUAccelerator):
            return HPUPrecisionPlugin(self._precision_flag)  # type: ignore
        if isinstance(self.accelerator, TPUAccelerator):
            if self._precision_flag == "32":
                return TPUPrecisionPlugin()
            elif self._precision_flag in ("16", "bf16"):
                if self._precision_flag == "16":
                    rank_zero_warn(
                        "You passed `Trainer(accelerator='tpu', precision=16)` but AMP"
                        " is not supported with TPUs. Using `precision='bf16'` instead."
                    )
                return TPUBf16PrecisionPlugin()

        if isinstance(self.strategy, ColossalAIStrategy):
            return ColossalAIPrecisionPlugin(self._precision_flag)
        if isinstance(self.strategy, DeepSpeedStrategy):
            return DeepSpeedPrecisionPlugin(self._precision_flag, self._amp_type_flag, self._amp_level_flag)

        if self._precision_flag == "32":
            return PrecisionPlugin()
        if self._precision_flag == "64":
            return DoublePrecisionPlugin()

        if self._precision_flag == "16" and self._accelerator_flag == "cpu":
            rank_zero_warn(
                "You passed `Trainer(accelerator='cpu', precision=16)` but native AMP is not supported on CPU."
                " Using `precision='bf16'` instead."
            )
            self._precision_flag = "bf16"

        if self._precision_flag in ("16", "bf16"):
            rank_zero_info(
                f"Using 16bit {self._amp_type_flag} Automatic Mixed Precision (AMP)"
                if self._precision_flag == "16"
                else "Using bfloat16 Automatic Mixed Precision (AMP)"
            )

            if self._amp_type_flag in (None, "native"):
                device = "cpu" if self._accelerator_flag == "cpu" else "cuda"

                if isinstance(self.strategy, (DDPShardedStrategy, DDPSpawnShardedStrategy)):
                    return ShardedNativeMixedPrecisionPlugin(self._precision_flag, device)
                if isinstance(self.strategy, DDPFullyShardedNativeStrategy):
                    return FullyShardedNativeNativeMixedPrecisionPlugin(self._precision_flag, device)
                if isinstance(self.strategy, DDPFullyShardedStrategy):
                    return FullyShardedNativeMixedPrecisionPlugin(self._precision_flag, device)
                return MixedPrecisionPlugin(self._precision_flag, device)

            if self._amp_type_flag == "apex":
                self._amp_level_flag = self._amp_level_flag or "O2"
                return ApexMixedPrecisionPlugin(self._amp_level_flag)

        raise RuntimeError("No precision set")

    def _validate_precision_choice(self) -> None:
        """Validate the combination of choices for precision, AMP type, and accelerator."""
        if isinstance(self.accelerator, TPUAccelerator):
            if self._precision_flag == "64":
                raise MisconfigurationException(
                    "`Trainer(accelerator='tpu', precision=64)` is not implemented."
                    " Please, open an issue in `https://github.com/Lightning-AI/lightning/issues`"
                    " requesting this feature."
                )
            if self._precision_plugin_flag and not isinstance(
                self._precision_plugin_flag, (TPUPrecisionPlugin, TPUBf16PrecisionPlugin)
            ):
                raise ValueError(
                    f"The `TPUAccelerator` can only be used with a `TPUPrecisionPlugin`,"
                    f" found: {self._precision_plugin_flag}."
                )
        if isinstance(self.accelerator, HPUAccelerator):
            if self._precision_flag not in ("16", "bf16", "32"):
                raise MisconfigurationException(
                    f"`Trainer(accelerator='hpu', precision={self._precision_flag!r})` is not supported."
                )
        if (
            self._precision_flag == "16"
            and isinstance(self.accelerator, CPUAccelerator)
            and self._amp_type_flag == "apex"
        ):
            raise MisconfigurationException(
                "You passed `Trainer(accelerator='cpu', precision=16, amp_type='apex')`"
                " but apex AMP not supported on CPU."
            )
        if self._precision_flag in ("16", "bf16") and self._amp_type_flag == "apex":
            if self._precision_flag == "bf16":
                raise MisconfigurationException(
                    "You passed `Trainer(amp_type='apex', precision='bf16')` but it's not supported."
                    " Remove the `amp_type` argument."
                )
            if isinstance(
                self.strategy,
                (DDPShardedStrategy, DDPSpawnShardedStrategy, DDPFullyShardedStrategy, DDPFullyShardedNativeStrategy),
            ):
                raise MisconfigurationException("Sharded plugins are not supported with apex.")

    def _lazy_init_strategy(self) -> None:
        """Lazily set missing attributes on the previously instantiated strategy."""
        self.strategy.accelerator = self.accelerator
        if self.precision_plugin:
            self.strategy.precision_plugin = self.precision_plugin
        if self.checkpoint_io:
            self.strategy.checkpoint_io = self.checkpoint_io
        if hasattr(self.strategy, "cluster_environment"):
            if self.strategy.cluster_environment is None:
                self.strategy.cluster_environment = self.cluster_environment
            self.cluster_environment = self.strategy.cluster_environment
        if hasattr(self.strategy, "parallel_devices"):
            if self.strategy.parallel_devices:
                self._parallel_devices = self.strategy.parallel_devices
            else:
                self.strategy.parallel_devices = self._parallel_devices
        if hasattr(self.strategy, "num_nodes"):
            self.strategy._num_nodes = self._num_nodes_flag
        if hasattr(self.strategy, "_layer_sync"):
            self.strategy._layer_sync = self._layer_sync
        if hasattr(self.strategy, "set_world_ranks"):
            self.strategy.set_world_ranks()
        self.strategy._configure_launcher()

        if _IS_INTERACTIVE and self.strategy.launcher and not self.strategy.launcher.is_interactive_compatible:
            raise MisconfigurationException(
                f"`Trainer(strategy={self.strategy.strategy_name!r})` is not compatible with an interactive"
                " environment. Run your code as a script, or choose one of the compatible strategies:"
                f" `Fabric(strategy=None|'dp'|'ddp_notebook')`."
                " In case you are spawning processes yourself, make sure to include the Trainer"
                " creation inside the worker function."
            )

        # TODO: should be moved to _check_strategy_and_fallback().
        # Current test check precision first, so keep this check here to meet error order
        if isinstance(self.accelerator, TPUAccelerator) and not isinstance(
            self.strategy, (SingleTPUStrategy, TPUSpawnStrategy)
        ):
            raise ValueError(
                "The `TPUAccelerator` can only be used with a `SingleTPUStrategy` or `TPUSpawnStrategy`,"
                f" found {self.strategy.__class__.__name__}."
            )

        if isinstance(self.accelerator, HPUAccelerator) and not isinstance(
            self.strategy, (SingleHPUStrategy, HPUParallelStrategy)
        ):
            raise ValueError(
                "The `HPUAccelerator` can only be used with a `SingleHPUStrategy` or `HPUParallelStrategy`,"
                f" found {self.strategy.__class__.__name__}."
            )

    @property
    def is_distributed(self) -> bool:
        # TODO: deprecate this property
        # Used for custom plugins.
        # Custom plugins should implement is_distributed property.
        if hasattr(self.strategy, "is_distributed") and not isinstance(self.accelerator, TPUAccelerator):
            return self.strategy.is_distributed
        distributed_strategy = (
            DDPStrategy,
            DDPSpawnShardedStrategy,
            DDPShardedStrategy,
            DDPFullyShardedNativeStrategy,
            DDPFullyShardedStrategy,
            DDPSpawnStrategy,
            DeepSpeedStrategy,
            TPUSpawnStrategy,
            HorovodStrategy,
            HPUParallelStrategy,
        )
        is_distributed = isinstance(self.strategy, distributed_strategy)
        if isinstance(self.accelerator, TPUAccelerator):
            is_distributed |= self.strategy.is_distributed
        return is_distributed
