# Copyright The PyTorch Lightning team.
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
from typing import List, Optional, Union

import torch

from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.accelerators.cpu import CPUAccelerator
from pytorch_lightning.accelerators.gpu import GPUAccelerator
from pytorch_lightning.accelerators.ipu import IPUAccelerator
from pytorch_lightning.accelerators.tpu import TPUAccelerator
from pytorch_lightning.plugins import (
    ApexMixedPrecisionPlugin,
    CheckpointIO,
    DeepSpeedPrecisionPlugin,
    DoublePrecisionPlugin,
    FullyShardedNativeMixedPrecisionPlugin,
    IPUPrecisionPlugin,
    NativeMixedPrecisionPlugin,
    PrecisionPlugin,
    ShardedNativeMixedPrecisionPlugin,
    TPUBf16PrecisionPlugin,
    TPUPrecisionPlugin,
)
from pytorch_lightning.plugins.environments import (
    BaguaEnvironment,
    ClusterEnvironment,
    KubeflowEnvironment,
    LightningEnvironment,
    LSFEnvironment,
    SLURMEnvironment,
    TorchElasticEnvironment,
)
from pytorch_lightning.strategies import (
    BaguaStrategy,
    DataParallelStrategy,
    DDP2Strategy,
    DDPFullyShardedStrategy,
    DDPShardedStrategy,
    DDPSpawnShardedStrategy,
    DDPSpawnStrategy,
    DDPStrategy,
    DeepSpeedStrategy,
    HorovodStrategy,
    ParallelStrategy,
    SingleDeviceStrategy,
    SingleTPUStrategy,
    Strategy,
    StrategyRegistry,
    TPUSpawnStrategy,
)
from pytorch_lightning.utilities import (
    _StrategyType,
    AMPType,
    device_parser,
    rank_zero_deprecation,
    rank_zero_info,
    rank_zero_warn,
)
from pytorch_lightning.utilities.enums import PrecisionType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _HOROVOD_AVAILABLE, _IPU_AVAILABLE, _TPU_AVAILABLE

log = logging.getLogger(__name__)

if _HOROVOD_AVAILABLE:
    import horovod.torch as hvd


class AcceleratorConnector:
    def __init__(
        self,
        devices,
        num_nodes,
        accelerator,  # reduce typing
        strategy: Optional[Union[str, Strategy]],
        plugins,
        precision,
        amp_type,
        amp_level,
        sync_batchnorm,
        benchmark,
        replace_sampler_ddp,
        deterministic: bool,
        num_processes,  # deprecated
        tpu_cores,  # deprecated
        ipus,  # deprecated
        gpus,  # deprecated
        gpu_ids,
    ):
        """
            A. accelerator flag could be:
                1. strategy class (deprecated in 1.5 will be removed in 1.7)
                2. strategy str (deprecated in 1.5 will be removed in 1.7)
                3. accelerator class
                4. accelerator str
                5. accelerator auto

            B. strategy flag could be :
                1. strategy class
                2. strategy str registered with strategyRegister
                3. strategy str in _strategy_type enum which listed in each strategy as
                   backend (registed these too, and _strategy_type could be deprecated)

            C. plugins flag could be:
                1. List of str, which could contains:
                    i. strategy str
                    ii. precision str (Not supported in the old accelerator_connector version)
                    iii. checkpoint_io str (Not supported in the old accelerator_connector version)
                    iv. cluster_environment str (Not supported in the old accelerator_connector version)
                2. List of class, which could contains:
                    i. strategy class (deprecated in 1.5 will be removed in 1.7)
                    ii. precision class (should be removed, and precision flag should allow user pass classes)
                    iii. checkpoint_io class
                    iv. cluster_environment class


        priorities which to take when:
            A. Class > str
            B. Strategy > Accelerator/precision/plugins
            C. When multiple flag set to the same thing? (ignore? not handled for now)

        """
        torch.backends.cudnn.benchmark = benchmark
        self.replace_sampler_ddp = replace_sampler_ddp
        self.sync_batchnorm = sync_batchnorm

        # --Parsing_flags------------------------------------------------------
        # Get registered strategies, existing accelerators and precision plugins
        self._existing_strategies_str = StrategyRegistry.available_strategies()
        # print(self._existing_strategies_str)
        self._existing_accelerator_type = ["tpu", "ipu", "gpu", "cpu"]
        self._supported_precision = PrecisionType.supported_types()

        # raise misconfig exceptions if their is conflict between flags
        # set the valid flag to self._x_flag after validation
        # for example: if accelerator is strategy class, set self._strategy_flag = accelerator
        # for devices: assign gpus ipus and etcs to accelerator_flag and devices_flag
        self._config_check_and_set_final_flags(strategy, accelerator, precision, plugins, amp_type, amp_level)
        self._device_config_check_and_set_final_flags(
            devices=devices, num_nodes=num_nodes, num_processes=num_processes, gpus=gpus, ipus=ipus, tpu_cores=tpu_cores
        )

        # --Accelerator-------------------------------------------------------------
        # handle `auto` and `None`
        if self._accelerator_flag == "auto" or self._accelerator_flag is None:
            self._choose_accelerator()
        # else:
        #     # [RFC] move to XAccelerator class init?
        #     self._check_device_availibility()
        self._set_parallel_devices_and_init_accelerator()

        # --Cluster_environment-----------------------------------------------------
        self._choose_and_init_cluster_environment()

        # --Strategy Part 1 : choose strategy and init strategy ---------------------------------------
        if self._strategy_flag is None:
            self._choose_strategy()
        # Reset strategy even user has specificed one
        self._strategy_check_and_fallbacks()
        self._init_strategy()

        # --Precision----------------------------------------------------------------
        self.precision_plugin = self._check_capatibility_and_init_precision()

        # --Strategy Part 2 : init Strategy and set Strategy properties -------------
        self._lazy_init_strategy()

    def _config_check_and_set_final_flags(self, strategy, accelerator, precision, plugins, amp_type, amp_level):
        """This method checks:

        1. strategy flag: strategy, accelerator and plugin can all set strategies
        2. accelerator: if accelerator flag is Accelerator related flag or class, set self._acceelrator_flag;
            If accelerator is strategy related, logic handled in 1 above
        3. precision could be set by precision and plugins flag
        4. plugins could be duplicated in strategy (handled by 1), precision (handled by 3),
            set checkpoint_io and cluster_environment
        """
        (
            self._strategy_flag,
            self._accelerator_flag,
            self._precision_flag,
            self._precision_plugin_flag,
            self._cluster_environment_flag,
            self.checkpoint_io,
            self._amp_level_flag,
            self._amp_type_flag,
        ) = (None, None, None, None, None, None, amp_type, amp_level)
        if plugins:
            plugins = [plugins] if not isinstance(plugins, list) else plugins

        if strategy:
            self._strategy_flag = strategy
            if strategy == "ddp_cpu":
                raise MisconfigurationException(
                    "`Trainer(strategy='ddp_cpu')` is not a valid strategy,"
                    " you can use `Trainer(strategy='ddp'|'ddp_spawn', accelerator='cpu')` instead."
                )
            if strategy == "tpu_spawn":
                raise MisconfigurationException(
                    "`Trainer(strategy='tpu_spawn')` is not a valid strategy,"
                    " you can use `Trainer(strategy='ddp_spawn', accelerator='tpu')` instead."
                )
            # handle duplications and conflict
            if isinstance(accelerator, Strategy) and strategy != accelerator:
                raise MisconfigurationException(
                    "strategy already set through strategy flag, but have also passed in through accelerator"
                )
            if (
                isinstance(accelerator, str)
                and accelerator in self._existing_strategies_str
                and strategy != accelerator
            ):
                raise MisconfigurationException(
                    "strategy str already set through strategy flag, but have also passed in through accelerator"
                )
            if plugins:
                for plugin in plugins:
                    if isinstance(plugin, Strategy):
                        raise MisconfigurationException(
                            f"You have passed `Trainer(strategy)`"
                            f" and you can only specify one strategy, but you have passed {plugin} as a plugin."
                        )
                    if isinstance(plugin, str) and plugin in self._existing_strategies_str:
                        raise MisconfigurationException(
                            f"You have passed `Trainer(strategy)`"
                            f" and you can only specify one strategy, but you have passed {plugin} as a plugin."
                        )

        if accelerator:
            if (
                accelerator in self._existing_accelerator_type
                or accelerator == "auto"
                or isinstance(accelerator, Accelerator)
            ):
                self._accelerator_flag = accelerator
            elif accelerator in self._existing_strategies_str or isinstance(accelerator, Strategy):
                rank_zero_deprecation(
                    f"Passing `Trainer(accelerator={accelerator!r})` has been deprecated"
                    f" in v1.5 and will be removed in v1.7. Use `Trainer(strategy={accelerator!r})` instead."
                )
                self._strategy_flag = accelerator
            elif accelerator == "ddp_cpu":
                rank_zero_warn(
                    "You requested one or more GPUs, but set `accelerator='ddp_cpu'`. Training will not use GPUs."
                )
                self._strategy_flag = accelerator

        if precision:
            if not PrecisionType.supported_type(precision):
                raise MisconfigurationException(
                    f"Precision {repr(precision)} is invalid. "
                    f"Allowed precision values: {PrecisionType.supported_types()}"
                )
            self._precision_flag = precision

        if plugins:
            for plugin in plugins:
                if isinstance(plugin, Strategy) or isinstance(plugin, str) and plugin in self._existing_strategies_str:
                    self._strategy_flag = plugin
                    rank_zero_deprecation(
                        f"Passing {plugin} `strategy` to the `plugins` flag in Trainer has been deprecated"
                        f" in v1.5 and will be removed in v1.7. Use `Trainer(strategy={plugin})` instead."
                    )

                elif isinstance(plugin, PrecisionPlugin):
                    self._precision_plugin_flag = plugin
                elif isinstance(plugin, str) and plugin in self._supported_precision:
                    self._precision_flag = plugin
                elif isinstance(plugin, CheckpointIO):
                    self.checkpoint_io = plugin
                elif isinstance(plugin, ClusterEnvironment):
                    self._cluster_environment_flag = plugin
                else:
                    raise MisconfigurationException(
                        f"Found invalid type for plugin {plugin}. Expected a precision or training type plugin."
                    )

        # if user pass in a strategy class which has accelerator, precision, checkpoint or cluster env set up
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
            # RFC existing accel_conn doesn't handle this, should we add conflict check?
            # eg: parallel_device is torch.device(cpu) but accelerator=gpu
            if hasattr(self._strategy_flag, "parallel_devices"):
                if self._strategy_flag.parallel_devices:
                    if self._strategy_flag.parallel_devices[0].type == "cpu":
                        self._accelerator_flag = "cpu"
                    if self._strategy_flag.parallel_devices[0].type == "cuda":
                        self._accelerator_flag = "gpu"

        amp_type = amp_type.lower() if isinstance(amp_type, str) else None
        self._amp_type_flag = AMPType.from_str(amp_type)

        if amp_level is not None and self._amp_type_flag != AMPType.APEX:
            raise MisconfigurationException(
                f"You have asked for `amp_level={amp_level!r}` but it's only supported with `amp_backend='apex'`."
            )
        self._amp_level_flag = amp_level

    def _device_config_check_and_set_final_flags(self, devices, num_nodes, num_processes, gpus, ipus, tpu_cores):
        if num_nodes == "auto":
            self._num_nodes_flag = 1
        else:
            self._num_nodes_flag = int(num_nodes) if num_nodes is not None else 1

        self._device_flag = devices
        # Delete when remove num_processes, gpus, ipus and tpu_cores
        self._gpus = gpus
        self._tpu_cores = tpu_cores
        gpus = device_parser.parse_gpu_ids(gpus)
        tpu_cores = device_parser.parse_tpu_cores(tpu_cores)
        deprecated_devices_specific_flag = num_processes or gpus or ipus or tpu_cores
        if deprecated_devices_specific_flag and deprecated_devices_specific_flag not in (0, "0"):
            self._mapping_deprecated_devices_specfic_info_to_accelerator_and_device_flag(
                devices, deprecated_devices_specific_flag, num_processes, gpus, ipus, tpu_cores
            )
        # Delete end
        if self._device_flag == "auto":
            if self._accelerator_flag is None:
                raise MisconfigurationException(
                    f"You passed `devices={devices}` but haven't specified"
                    " `accelerator=('auto'|'tpu'|'gpu'|'ipu'|'cpu')` for the devices mapping"
                )

    def _mapping_deprecated_devices_specfic_info_to_accelerator_and_device_flag(
        self, devices, deprecated_devices_specific_flag, num_processes, gpus, ipus, tpu_cores
    ):
        # set devices base on num_processes, gpus, ipus, tpu_cores
        if devices:
            rank_zero_warn(
                f"The flag `devices={devices}` will be ignored, "
                f"instand the device specific number {deprecated_devices_specific_flag} will be used"
            )

        if [(num_processes is not None), (gpus is not None), (ipus is not None), (tpu_cores is not None)].count(
            True
        ) > 1:
            rank_zero_warn("more than one device specifc flag has been set")
        self._device_flag = deprecated_devices_specific_flag

        if not self._accelerator_flag:
            # set accelerator type base on num_processes, gpus, ipus, tpu_cores
            if ipus:
                self._accelerator_flag = "ipu"
            if tpu_cores:
                self._accelerator_flag = "tpu"
            if gpus:
                self._accelerator_flag = "gpu"
            if num_processes:
                self._accelerator_flag = "cpu"

    def _choose_accelerator(self):
        if self._accelerator_flag == "auto":
            if _TPU_AVAILABLE:
                self._accelerator_flag = "tpu"
            elif _IPU_AVAILABLE:
                self._accelerator_flag = "ipu"
            elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self._accelerator_flag = "gpu"
            else:
                self._accelerator_flag = "cpu"
                if self._device_flag == "auto":
                    self._device_flag = 1
        # [RFC] this is current logic, if accelerator not set, default cpu?
        else:
            self._accelerator_flag = "cpu"

    # TODO move this to xAccelerator
    # def _check_device_availibility(self):
    #     for accelerator_flag, available in zip(
    #         self._existing_accelerator_type, [_TPU_AVAILABLE, _IPU_AVAILABLE, torch.cuda.is_available(), True]
    #     ):
    #         # only apply to gpu to keep backward compatibility
    #         if self._accelerator_flag == accelerator_flag:
    #             if not available:
    #                 raise MisconfigurationException(
    #                     f"You choice {accelerator_flag} accelerator, but {accelerator_flag} is not available"
    #                 )

    def _set_parallel_devices_and_init_accelerator(self):
        self._parallel_devices = []
        if isinstance(self._accelerator_flag, Accelerator):
            self.accelerator = self._accelerator_flag
        elif self._accelerator_flag == "tpu":
            self.accelerator = TPUAccelerator()
            if self._device_flag == "auto" or not self._device_flag:
                self._device_flag = TPUAccelerator.auto_device_count()
            if isinstance(self._device_flag, int):
                self._parallel_devices = list(range(self._device_flag))
            else:
                self._parallel_devices = self._device_flag

        elif self._accelerator_flag == "ipu":
            self.accelerator = IPUAccelerator()
            if self._device_flag == "auto" or not self._device_flag:
                self._device_flag = IPUAccelerator.auto_device_count()
            if isinstance(self._device_flag, int):
                self._parallel_devices = list(range(self._device_flag))

        elif self._accelerator_flag == "gpu":
            self.accelerator = GPUAccelerator()
            if self._device_flag == "auto" or not self._device_flag:
                self._device_flag = GPUAccelerator.auto_device_count()
            if isinstance(self._device_flag, int) or isinstance(self._device_flag, str):
                self._device_flag = int(self._device_flag)
                self._parallel_devices = (
                    [torch.device("cuda", i) for i in device_parser.parse_gpu_ids(self._device_flag)]
                    if self._device_flag != 0
                    else []
                )
            else:
                self._parallel_devices = [torch.device("cuda", i) for i in self._device_flag]

        elif self._accelerator_flag == "cpu":
            self.accelerator = CPUAccelerator()
            if self._device_flag == "auto" or not self._device_flag:
                self._device_flag = CPUAccelerator.auto_device_count()
            if isinstance(self._device_flag, int):
                self._parallel_devices = [torch.device("cpu")] * self._device_flag
            else:
                rank_zero_warn(
                    "The flag `devices` must be an int with `accelerator='cpu'`,"
                    f" got `devices={self._device_flag}` instead."
                )

        self._gpus = self._device_flag if not self._gpus else self._gpus

    def _choose_and_init_cluster_environment(self):
        self.cluster_environment = LightningEnvironment()
        if isinstance(self._cluster_environment_flag, ClusterEnvironment):
            self.cluster_environment = self._cluster_environment_flag
        elif self._is_slurm_managing_tasks():
            rank_zero_info("Multiprocessing is handled by SLURM.")
            self.cluster_environment = SLURMEnvironment()
        else:
            for env_type in (TorchElasticEnvironment, KubeflowEnvironment, LSFEnvironment):
                if env_type.detect():
                    self.cluster_environment = env_type()


    @property
    def _is_sharded_training_type(self) -> bool:
        return isinstance(self._strategy, (DDPShardedStrategy, DDPSpawnShardedStrategy))

    def _is_slurm_managing_tasks(self):
        """used by choosing cluster enviroment."""
        if not SLURMEnvironment.detect() or SLURMEnvironment.job_name() == "bash":
            return False

        total_requested_devices = len(self._parallel_devices) * self._num_nodes_flag
        num_slurm_tasks = int(os.environ["SLURM_NTASKS"], 0)
        return num_slurm_tasks == total_requested_devices

    def _choose_strategy(self):
        if self._accelerator_flag == "ipu":
            self._strategy_flag = "ipu"
        elif self._accelerator_flag == "tpu":
            if self._parallel_devices and len(self._parallel_devices) > 1:
                self._strategy_flag = "tpu_spawn"
            else:
                # TODO lazy initialized device, then here could be self._strategy_flag = "single_tpu_device"
                self._strategy_flag = SingleTPUStrategy(device=self._parallel_devices[0])
        elif _HOROVOD_AVAILABLE and ("OMPI_COMM_WORLD_RANK" in os.environ or "HOROVOD_RANK" in os.environ):
            self._strategy_flag = "horovod"
        else:
            if self._num_nodes_flag > 1:
                self._strategy_flag = "ddp"
            elif len(self._parallel_devices) <= 1:
                device = (
                    device_parser.determine_root_gpu_device(self._parallel_devices)
                    if self._accelerator_flag == "gpu"
                    else "cpu"
                )
                # TODO lazy initialized device, then here could be self._strategy_flag = "single_device"
                self._strategy_flag = SingleDeviceStrategy(device=device)
            elif len(self._parallel_devices) > 1:
                self._strategy_flag = "ddp_spawn"
            else:
                self._strategy_flag = "ddp"

    def _strategy_check_and_fallbacks(self):
        # current logic, fallback only apply to user pass in str config not object config
        _strategy_flag = "" if isinstance(self._strategy_flag, Strategy) else self._strategy_flag

        if _strategy_flag == "ddp_cpu":
            if _TPU_AVAILABLE:
                raise MisconfigurationException(
                    "`accelerator='ddp_cpu'` is not supported on TPU machines. "
                    "Learn more: https://github.com/PyTorchLightning/pytorch-lightning/issues/7810"
                )
            if self._device_flag == 1 and self._num_nodes_flag > 1:
                _strategy_flag = "ddp"
            else:
                _strategy_flag = "ddp_spawn"
            if self._accelerator_flag == "gpu":
                rank_zero_warn(
                    "You requested one or more GPUs, but set `accelerator='ddp_cpu'`. Training will not use GPUs."
                )
        if _strategy_flag in ("ddp_spawn", "ddp_spawn_find_unused_parameters_false") and (
            TorchElasticEnvironment.detect() or KubeflowEnvironment.detect() or self._is_slurm_managing_tasks()
        ):
            _strategy_flag = "ddp"
        if _strategy_flag in ("dp", "ddp2") and self._accelerator_flag == "cpu":
            rank_zero_warn(f"{_strategy_flag!r} is not supported on CPUs, hence setting `strategy='ddp'`.")
            _strategy_flag = "ddp"

        if _strategy_flag:
            self._strategy_flag = _strategy_flag

    def handle_horovod(self):
        if self._num_nodes_flag > 1:
            raise MisconfigurationException(
                "Horovod does not support setting num_nodes / num_gpus explicitly. Use "
                "horovodrun / mpirun to configure the number of processes."
            )

        if not _HOROVOD_AVAILABLE:
            raise MisconfigurationException(
                'Requested `accelerator="horovod"`, but Horovod is not installed.'
                "Install with \n $HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]"
            )

        hvd.init()
        if isinstance(self.accelerator, GPUAccelerator):
            # Horovod assigns one local GPU per process
            self._parallel_devices = list(range(hvd.local_size()))
        else:
            self._parallel_devices = [torch.device("cpu")] * hvd.local_size()

    def _init_strategy(self):
        if isinstance(self._strategy_flag, HorovodStrategy) or self._strategy_flag == "horovod":
            # handle horovod has to happen before initialize strategy because HorovodStrategy needs hvd.init() first.
            # TODO lazy initialized and setup horovod strategy `global_rank`
            self.handle_horovod()
        if isinstance(self._strategy_flag, str):
            self.strategy = StrategyRegistry.get(self._strategy_flag)
        elif isinstance(self._strategy_flag, Strategy):
            self.strategy = self._strategy_flag
        else:
            raise RuntimeError(f"{self.strategy} is not valid type: {self.strategy}")

    def _check_capatibility_and_init_precision(self):
        self._precision_misconfig_check()
        if isinstance(self._precision_plugin_flag, PrecisionPlugin):
            return self._precision_plugin_flag

        if isinstance(self.accelerator, IPUAccelerator):
            return IPUPrecisionPlugin(self._precision_flag)
        if isinstance(self.accelerator, TPUAccelerator):
            if self._precision_flag == 32:
                return TPUPrecisionPlugin()
            elif self._precision_flag in (16, "bf16"):
                if self._precision_flag == 16:
                    rank_zero_warn(
                        f"You passed `Trainer(accelerator='tpu', precision=16)` but {self._amp_type_flag.value} AMP"
                        f" is not supported with TPUs. Using `precision='bf16'` instead."
                    )
                return TPUBf16PrecisionPlugin()
        if self._strategy_flag == "deepspeed" or isinstance(self._strategy_flag, DeepSpeedStrategy):
            return DeepSpeedPrecisionPlugin(self._precision_flag, self._amp_type_flag, self._amp_level_flag)

        if self._precision_flag == 32:
            return PrecisionPlugin()
        if self._precision_flag == 64:
            return DoublePrecisionPlugin()

        if self._precision_flag == 16 and self._accelerator_flag == "cpu":
            rank_zero_warn(
                "You passed `Trainer(accelerator='cpu', precision=16)` but native AMP is not supported on CPU."
                " Using `precision='bf16'` instead."
            )
            self._precision_flag = "bf16"

        if self._precision_flag in (16, "bf16"):
            rank_zero_info(
                f"Using 16bit {self._amp_type_flag.value} Automatic Mixed Precision (AMP)"
                if self._precision_flag == 16
                else "Using bfloat16 Automatic Mixed Precision (AMP)"
            )

            if self._amp_type_flag == AMPType.NATIVE:
                device = "cpu" if self._accelerator_flag == "cpu" else "cuda"

                if isinstance(self.strategy, (DDPShardedStrategy, DDPSpawnShardedStrategy)):
                    return ShardedNativeMixedPrecisionPlugin(self._precision_flag, device)
                if isinstance(self.strategy, DDPFullyShardedStrategy):
                    return FullyShardedNativeMixedPrecisionPlugin(self._precision_flag, device)
                return NativeMixedPrecisionPlugin(self._precision_flag, device)

            if self._amp_type_flag == AMPType.APEX:
                self._amp_level_flag = self._amp_level_flag or "O2"
                return ApexMixedPrecisionPlugin(self._amp_level_flag)

        raise RuntimeError("No precision set")

    def _precision_misconfig_check(self):
        # TODO change exception type to ImpactableConfigurationException
        if isinstance(self.accelerator, IPUAccelerator):
            if self._precision_flag not in (16, 32):
                raise MisconfigurationException(
                    f"`Trainer(accelerator='ipu', precision={self._precision_flag!r})` is not supported."
                )
        if isinstance(self.accelerator, TPUAccelerator) and self._precision_flag == 64:
            raise MisconfigurationException(
                "`Trainer(accelerator='tpu', precision=64)` is not implemented."
                " Please, open an issue in `https://github.com/PyTorchLightning/pytorch-lightning/issues`"
                " requesting this feature."
            )
        if (
            isinstance(self.accelerator, TPUAccelerator)
            and self._precision_plugin_flag
            and not isinstance(self._precision_plugin_flag, (TPUPrecisionPlugin, TPUBf16PrecisionPlugin))
        ):
            raise ValueError(
                f"The `TPUAccelerator` can only be used with a `TPUPrecisionPlugin`,"
                f" found: {self._precision_plugin_flag}."
            )
        if (
            self._precision_flag == 16
            and isinstance(self.accelerator, CPUAccelerator)
            and self._amp_type_flag == AMPType.APEX
        ):
            raise MisconfigurationException(
                "You passed `Trainer(accelerator='cpu', precision=16, amp_type='apex')`"
                " but apex AMP not supported on CPU."
            )
        if self._precision_flag == "bf16" and self._amp_type_flag != AMPType.NATIVE:
            raise MisconfigurationException(
                f"You passed `Trainer(amp_type={self._amp_type_flag.value!r}, precision='bf16')` but "
                "it's not supported. Try using `amp_type='native'` instead."
            )
        if self._precision_flag in (16, "bf16") and self._amp_type_flag == AMPType.APEX:
            if isinstance(self.strategy, (DDPShardedStrategy, DDPSpawnShardedStrategy, DDPFullyShardedStrategy)):
                raise MisconfigurationException(
                    "Sharded plugins are not supported with apex, please switch to `amp_backend='native'`."
                )

    def _lazy_init_strategy(self):
        # set strategy properties
        self.strategy.accelerator = self.accelerator
        if self.precision_plugin:
            self.strategy.precision_plugin = self.precision_plugin
        if self.checkpoint_io:
            self.strategy.checkpoint_io = self.checkpoint_io
        if hasattr(self.strategy, "cluster_environment"):
            self.strategy.cluster_environment = self.cluster_environment
        if hasattr(self.strategy, "parallel_devices"):
            if self.strategy.parallel_devices:
                self._parallel_devices = self.strategy.parallel_devices
            else:
                self.strategy.parallel_devices = self._parallel_devices
        if hasattr(self.strategy, "num_nodes"):
            self.strategy._num_nodes = self._num_nodes_flag
        if hasattr(self.strategy, "sync_batchnorm"):
            self.strategy.sync_batchnorm = self.sync_batchnorm
        if hasattr(self.strategy, "set_world_ranks"):
            self.strategy.set_world_ranks()

        from pytorch_lightning.utilities import _IS_INTERACTIVE

        interactive_compatible_strategy = ("dp", "ddp_spawn", "ddp_sharded_spawn", "tpu_spawn")
        if _IS_INTERACTIVE and self.strategy.distributed_backend not in interactive_compatible_strategy:
            raise MisconfigurationException(
                f"`Trainer(strategy={self.strategy.distributed_backend!r})` or"
                f" `Trainer(accelerator={self.strategy.distributed_backend!r})` is not compatible with an interactive"
                " environment. Run your code as a script, or choose one of the compatible backends:"
                f" {', '.join(interactive_compatible_strategy)}."
                " In case you are spawning processes yourself, make sure to include the Trainer"
                " creation inside the worker function."
            )

        # TODO should be moved to _strategy_check_and_fallbacks().
        # Current test check precision first, so keep this check here to meet error order
        if isinstance(self.accelerator, TPUAccelerator) and not isinstance(
            self.strategy, (SingleTPUStrategy, TPUSpawnStrategy)
        ):
            raise ValueError(
                "The `TPUAccelerator` can only be used with a `SingleTPUStrategy` or `TPUSpawnStrategy`,"
                f" found {self.strategy}."
            )

    ##############################################################################
    # the following logic should be deprecated/removed, and these information should be
    # retrive from strategies and accelerators
    # Added here to keep backward compabilities

    @property
    def parallel_devices(self) -> List[Union[torch.device, int]]:
        return self._parallel_devices

    # def _distrib_type():
    @property
    def device_type(self):
        if isinstance(self.accelerator, CPUAccelerator):
            return "cpu"
        if isinstance(self.accelerator, GPUAccelerator):
            return "gpu"
        if isinstance(self.accelerator, TPUAccelerator):
            return "tpu"
        if isinstance(self.accelerator, IPUAccelerator):
            return "ipu"

    @property
    def num_nodes(self):
        return self._num_nodes_flag

    @property
    def num_processes(self):
        return self.devices if self.devices is not None else 1

    @property
    def root_gpu(self) -> Optional[int]:
        return (
            self.strategy.root_device.index
            if not isinstance(self.accelerator, (IPUAccelerator, TPUAccelerator))
            else None
        )

    @property
    def devices(self):
        if isinstance(self.strategy, SingleDeviceStrategy):
            return 1
        elif isinstance(self.strategy, ParallelStrategy):
            return len(self.strategy.parallel_devices)
        else:
            return 0

    @property
    def tpu_cores(self) -> int:
        return self.devices

    @property
    def ipus(self) -> int:
        return self.devices

    @property
    def num_gpus(self) -> int:
        if isinstance(self.accelerator, GPUAccelerator):
            return self.devices
        else:
            return 0

    # def parallel_device_ids():
    @property
    def gpus(self):
        return self._gpus
        # if isinstance(self.accelerator, GPUAccelerator) else 0

    @property
    def parallel_device_ids(self):
        return [i for i in range(len(self.parallel_devices))] if isinstance(self.accelerator, GPUAccelerator) else None

    @property
    def is_distributed(self):
        # Used for custom plugins.
        # Custom plugins should implement is_distributed property.
        if hasattr(self.strategy, "is_distributed") and not isinstance(self.accelerator, TPUAccelerator):
            return self.strategy.is_distributed
        distributed_strategy = (
            DDP2Strategy,
            DDPStrategy,
            DDPSpawnShardedStrategy,
            DDPShardedStrategy,
            DDPFullyShardedStrategy,
            DDPSpawnStrategy,
            DeepSpeedStrategy,
            TPUSpawnStrategy,
            HorovodStrategy,
        )
        is_distributed = isinstance(self.strategy, distributed_strategy)
        if isinstance(self.accelerator, TPUAccelerator):
            is_distributed |= self.strategy.is_distributed
        return is_distributed

    @property
    def has_ipu(self):
        return isinstance(self.accelerator, IPUAccelerator)

    @property
    def use_ipu(self):
        return self.has_ipu

    @property
    def has_tpu(self):
        return isinstance(self.accelerator, TPUAccelerator)

    @property
    def use_dp(self):
        return isinstance(self.strategy, DataParallelStrategy)

    @property
    def _strategy_type(self) -> _StrategyType:
        return self.strategy.distributed_backend
