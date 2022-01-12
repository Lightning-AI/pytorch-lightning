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
from typing import List, Optional, Sequence, Union
from weakref import proxy

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
    ClusterEnvironment,
    KubeflowEnvironment,
    LightningEnvironment,
    LSFEnvironment,
    SLURMEnvironment,
    TorchElasticEnvironment,
)
from pytorch_lightning.strategies import (
    DataParallelStrategy,
    DDP2Strategy,
    DDPFullyShardedStrategy,
    DDPShardedStrategy,
    DDPSpawnShardedStrategy,
    DDPSpawnStrategy,
    DDPStrategy,
    DeepSpeedStrategy,
    HorovodStrategy,
    IPUStrategy,
    SingleDeviceStrategy,
    SingleTPUStrategy,
    Strategy,
    StrategyRegistry,
    TPUSpawnStrategy,
)
from pytorch_lightning.utilities import (
    _AcceleratorType,
    _StrategyType,
    AMPType,
    device_parser,
    rank_zero_deprecation,
    rank_zero_info,
    rank_zero_warn,
)
from pytorch_lightning.utilities.enums import PrecisionType
from pytorch_lightning.utilities.exceptions import MisconfigurationException, DeviceNotAvailibleException, ImpactableConfigurationException
from pytorch_lightning.utilities.imports import (
    _HOROVOD_AVAILABLE,
    _IPU_AVAILABLE,
    _GPU_AVAILABLE,
    _TORCH_GREATER_EQUAL_1_8,
    _TPU_AVAILABLE,
)

if _HOROVOD_AVAILABLE:
    import horovod.torch as hvd

log = logging.getLogger(__name__)


class AcceleratorConnector:
    def __init__(
        self,
        devices,
        num_nodes,
        accelerator, # reduce typing
        strategy: Optional[Union[str, Strategy]],
        plugins,
        precision,
        amp_type,
        amp_level,
        sync_batchnorm,
        benchmark,
        replace_sampler_ddp,
        deterministic: bool,
        num_processes, # deprecated
        tpu_cores, # deprecated
        ipus, # deprecated
        gpus, # deprecated
        gpu_ids,
    ):
        """
            A. accelerator could be:
                1. strategy class (deprecated in 1.5 will be removed in 1.7)
                2. strategy str (deprecated in 1.5 will be removed in 1.7)
                3. accelerator class
                4. accelerator str
                5. accelerator auto

            B. strategy could be :
                1. strategy class
                2. strategy str registered with strategyRegister
                3. strategy str in _strategy_type enum which listed in each strategy as backend (registed these too, and _strategy_type could be deprecated)

            C. plugins could be:
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

        # Get registered strategies, existing accelerators and precision plugins
        self._existing_strategies_str = StrategyRegistry.available_strategies()
        print(self._existing_strategies_str)
        self._existing_accelerator_type = ["tpu", "ipu", "gpu", "cpu"]
        self._supported_precision = PrecisionType.supported_types()

        # raise misconfig exceptions if their is conflict between flags
        # set the valid flag to self._x_flag after validation
        # for example: if accelerator is strategy class, set self._strategy_flag = accelerator
        # for devices: assign gpus ipus and etcs to accelerator_flag and devices_flag
        self._config_check_and_set_final_flags(strategy, accelerator, precision, plugins, amp_type, amp_level)
        self._device_config_check_and_set_final_flags(devices=devices, num_nodes=num_nodes, num_processes=num_processes, gpus=gpus, ipus=ipus, tpu_cores=tpu_cores)

        # handle auto and choose flag when user hasn't set it up.
        if self._accelerator_flag == 'auto' or self._accelerator_flag is None:
            self._choose_accelerator()
        else:
            # [RFC] move to XAccelerator class init?
            self._check_device_availibility()

        # Accelerator initialization
        # TODO devices logic handling still in process, not ready for reviews
        self._set_parallel_devices_and_init_accelerator()

        # handle strategy flag is not set, choose for user
        if self._strategy_flag is None:
            self._choose_strategy()

        self._choose_and_init_cluster_environment()
        self._check_capatibility_and_init_precision()
        self._init_strategy()


    def _config_check_and_set_final_flags(self, strategy, accelerator, precision, plugins, amp_type, amp_level):
        """
        This method checks:
            1. strategy flag: strategy, accelerator and plugin can all set strategies
            2. accelerator: if accelerator flag is Accelerator related flag or class, set self._acceelrator_flag;
                If accelerator is strategy related, logic handled in 1 above
            3. precision could be set by precision and plugins flag
            4. plugins could be duplicated in strategy (handled by 1), precision (handled by 3), set checkpoint_io and cluster_environment
        """
        self._strategy_flag, self._accelerator_flag, self._precision_flag, self._cluster_environment, self.checkpoint_io, self._amp_level_flag, self._amp_type_flag = None, None, None, None, None, amp_type, amp_level
        if strategy:
            self._strategy_flag = strategy
            # handle duplications and conflict
            if isinstance(accelerator, Strategy) and strategy != accelerator:
                raise MisconfigurationException("strategy already set through strategy flag, duplicated in accelerator")
            if isinstance(accelerator, str) and accelerator in self._existing_strategies_str and strategy != accelerator:
                raise MisconfigurationException("strategy str already set through strategy flag, duplicated in accelerator")
            if plugins:
                for plugin in plugins:
                    if isinstance(plugin, Strategy) and strategy != plugin:
                        raise MisconfigurationException("strategy already set through strategy flag, duplicated in plugins")
                    if isinstance(plugin, str) and plugin in self._existing_strategies_str:
                        raise MisconfigurationException("strategy already set through strategy flag, duplicated in plugins")


        if accelerator in self._existing_accelerator_type or accelerator=="auto" or isinstance(accelerator, Accelerator):
            self._accelerator_flag = accelerator
        elif accelerator in self._existing_strategies_str or isinstance(accelerator, Strategy):
            rank_zero_deprecation(
                f"Passing `Trainer(accelerator={accelerator!r})` has been deprecated"
                f" in v1.5 and will be removed in v1.7. Use `Trainer(strategy={accelerator!r})` instead."
            )
            self._strategy_flag = accelerator


        if precision:
            self._precision_flag = precision
            # handle duplications and conflict
            if plugins:
                for plugin in plugins:
                    if isinstance(plugin, PrecisionPlugin):
                        raise MisconfigurationException("precision set in both precision flag and plugin flag")

        if plugins:
            for plugin in plugins:
                if isinstance(plugin, Strategy) or isinstance(plugin, str) and plugin in self._existing_strategies_str:
                    self._strategy_flag = plugin
                elif isinstance(plugin, PrecisionPlugin) or isinstance(plugin, str) and plugin in self._supported_precision:
                    self._precision_flag = plugin
                elif isinstance(plugin, CheckpointIO):
                    self.checkpoint_io =  plugin
                elif isinstance(plugin, ClusterEnvironment):
                    self._cluster_environment = plugin
                else:
                    raise MisconfigurationException(f"Does not recognize flag {plugin}")


        # if user pass in a strategy class which has accelerator, precision, checkpoint or cluster env set up
        if self._strategy_flag and isinstance(self._strategy_flag, Strategy):
            if self._strategy_flag.accelerator:
                if self._accelerator_flag:
                    raise MisconfigurationException("accelerator set through both strategy class and accelerator flag, choose one")
                else:
                    self._accelerator_flag = self._strategy_flag.accelerator
            if self._strategy_flag.precision_plugin:
                # precision has default value 32, we can not tell whether user set it or not [RFC] remove default from trainer?
                # if self._precision_flag:
                #     raise MisconfigurationException("precision set through both strategy class and flags, choose one place to set")
                # else:
                self._precision_flag = self._strategy_flag.precision_plugin
            if self._strategy_flag.checkpoint_io:
                if self.checkpoint_io:
                    raise MisconfigurationException("checkpoint_io set through both strategy class and plugins, choose one")
                else:
                    self.checkpoint_io = self._strategy_flag.checkpoint_io
            if getattr(self._strategy_flag, "cluster_environment", None):
                if self._cluster_environment:
                    raise MisconfigurationException("cluster_environment set through both strategy class and plugins, choose one")
                else:
                    self._cluster_environment = getattr(self._strategy_flag, "cluster_environment")


        amp_type = amp_type.lower() if isinstance(amp_type, str) else None
        self._amp_type_flag = AMPType.from_str(amp_type) if amp_type is not None else None

        # TODO still working on these flags
        # if amp_level is not None and self._amp_type_flag != AMPType.APEX:
        #     raise MisconfigurationException(
        #         f"You have asked for `amp_level={self._amp_level_flag!r}` but it's only supported with `amp_backend='apex'`."
        #     )
        self._amp_level_flag = amp_level


    def _device_config_check_and_set_final_flags(self, devices, num_nodes, num_processes, gpus, ipus, tpu_cores):
        if num_nodes == "auto":
            self._num_nodes_flag = 1
        else :
            self._num_nodes_flag = int(num_nodes) if num_nodes is not None else 1

        ##### to be deleted v1.7
        deprecated_devices_specific_nums = num_processes or gpus or ipus or tpu_cores
        self._mapping_deprecated_devices_specfic_info_to_accelerator_and_device_flag(devices, deprecated_devices_specific_nums, num_processes, gpus, ipus, tpu_cores)
        ##### deleted end
        if devices == "auto":
            if self._accelerator_flag is None:
                raise MisconfigurationException(
                    f"You passed `devices={devices}` but haven't specified"
                    " `accelerator=('auto'|'tpu'|'gpu'|'ipu'|'cpu')` for the devices mapping"
                )
        if not self._device_flag:
            self._device_flag = devices



    def _mapping_deprecated_devices_specfic_info_to_accelerator_and_device_flag(self, devices, deprecated_devices_specific_nums, num_processes, gpus, ipus, tpu_cores):
        ##### to be deleted v1.7vbg
        # set devices base on num_processes, gpus, ipus, tpu_cores
        if devices:
            rank_zero_warn(f"will be ignored, instand the device specific number {deprecated_devices_specific_nums} will be used")
        if [(num_processes is not None), (gpus is not None), (ipus is not None), (tpu_cores is not None)].count(True) > 1:
            rank_zero_warn(f"more than one device specifc flag has been set")
        self._device_flag = deprecated_devices_specific_nums

        if not self._accelerator_flag:
        # set accelerator type base on num_processes, gpus, ipus, tpu_cores
            if num_processes:
                self._accelerator_flag = "cpu"
            if gpus:
                self._accelerator_flag = "gpu"
            if tpu_cores:
                self._accelerator_flag = "tpu"
            if ipus:
                self._accelerator_flag = "ipu"
        #### delete end

    def _choose_accelerator(self):
        if self._accelerator_flag == "auto":
            if _TPU_AVAILABLE:
                self._accelerator_flag = "tpu"
            elif _IPU_AVAILABLE:
                self._accelerator_flag = "ipu"
            elif _GPU_AVAILABLE:
                self._accelerator_flag = "gpu"
            else:
                self._accelerator_flag = "cpu"
        # [RFC] this is current logic, if accelerator not set, default cpu?
        else:
            self._accelerator_flag = "cpu"


    def _check_device_availibility(self):
        for accelerator_flag, available in zip(self._existing_accelerator_type, [_TPU_AVAILABLE, _IPU_AVAILABLE, _GPU_AVAILABLE, True]):
            if self._accelerator_flag == accelerator_flag:
                if not available:
                    raise DeviceNotAvailibleException(f"{accelerator_flag} not avalible")

    # TODO in progress for setting up devices
    def _set_parallel_devices_and_init_accelerator(self):
        self._parallel_devices = []

        if isinstance(self._accelerator_flag, Accelerator):
            self.accelerator = self._accelerator_flag()
        elif self._accelerator_flag == "tpu":
            self.accelerator = TPUAccelerator()
            if self._device_flag == "auto" or not self._device_flag:
                self._device_flag = TPUAccelerator.auto_device_count()
            if isinstance(self._device_flag, int):
                self._parallel_devices = list(range(self._device_flag))

        elif self._accelerator_flag == "ipu":
            self.accelerator = IPUAccelerator()
            if self._device_flag == "auto" or not self._device_flag:
                self._device_flag = IPUAccelerator.auto_device_count()
            if isinstance(self._device_flag, int):
                self._parallel_devices = list(range(self._device_flag))

        elif self._accelerator_flag == "gpu":
            self.accelerator = GPUAccelerator()
            if self._device_flag == "auto" or not self._device_flag:
                self._device_flag =  GPUAccelerator.auto_device_count()
            if isinstance(self._device_flag, int):
                self._parallel_devices = [torch.device("cuda", i) for i in device_parser.parse_gpu_ids(self._device_flag)]

        elif self._accelerator_flag == "cpu":
            self.accelerator = CPUAccelerator()
            if self._device_flag == "auto" or not self._device_flag:
                self._device_flag =  CPUAccelerator.auto_device_count()
            if isinstance(self._device_flag, int):
                self._parallel_devices = [torch.device("cpu")] * self._device_flag


    def _choose_and_init_cluster_environment(self):
        self.cluster_environment = LightningEnvironment()
        if isinstance(self._cluster_environment, ClusterEnvironment):
            self.cluster_environment = self._cluster_environment
        elif self._is_slurm_managing_tasks():
            rank_zero_info("Multiprocessing is handled by SLURM.")
            self.cluster_environment = SLURMEnvironment()
        else:
            for env_type in (TorchElasticEnvironment, KubeflowEnvironment, LSFEnvironment):
                if env_type.detect():
                    self.cluster_environment = env_type()


    def _is_slurm_managing_tasks(self):
        """
            used by choosing cluster enviroment
        """
        if (
            (not self._strategy_flag=="ddp" and not self._strategy_flag=="ddp2")
            or not SLURMEnvironment.detect()
            or SLURMEnvironment.job_name() == "bash"  # in interactive mode we don't manage tasks
        ):
            return False

        total_requested_devices = len(self._parallel_devices) * self._num_nodes_flag
        num_slurm_tasks = int(os.environ["SLURM_NTASKS"], 0)
        return num_slurm_tasks == total_requested_devices

    def _choose_strategy(self):
        if _HOROVOD_AVAILABLE and ("OMPI_COMM_WORLD_RANK" in os.environ or "HOROVOD_RANK" in os.environ):
            self._strategy_flag = HorovodStrategy()

        if self._accelerator_flag == "ipu":
            self._strategy_flag = IPUStrategy()
        elif self._accelerator_flag == "tpu":
            if self._parallel_devices and len(self._parallel_devices)>1:
                self._strategy_flag = TPUSpawnStrategy()
            else:
                self._srategy_flag = SingleTPUStrategy()

        # [RFC] in existing logic SingleDevice strategy choice diverge between cpu and gpu, should we merge?
        elif self._accelerator_flag == "gpu":
            if self._num_nodes_flag > 1:
                self._strategy_flag = DDPStrategy()
            elif len(self._parallel_devices) == 1:
                self._strategy_flag = DDPStrategy()
            elif len(self._parallel_devices) > 1:
                self._strategy_flag = DDPSpawnStrategy()
            else:
                self._strategy_flag = DDPStrategy()
        else:
            if self._num_nodes_flag > 1:
                self._strategy_flag = DDPStrategy()
            elif len(self._parallel_devices) <= 1:
                device = torch.device("cuda") if self._accelerator_flag == "gpu" else "cpu"
                self._strategy_flag = SingleDeviceStrategy(device = device)
            elif len(self._parallel_devices) > 1:
                self._strategy_flag = DDPSpawnStrategy()
            else:
                self._strategy_flag = DDPStrategy()


    def _check_capatibility_and_init_precision(self):
        self._precision_misconfig_check()
        if isinstance(self._precision_flag, PrecisionPlugin):
            self.precision_plugin = self._precision_flag

        if self._accelerator_flag =="ipu":
            self.precision_plugin = IPUPrecisionPlugin(self._precision_flag)
        if self._accelerator_flag == "tpu":
            if self._precision_flag == 32:
                self.precision_plugin = TPUPrecisionPlugin()
            elif self._precision_flag in (16, "bf16"):
                if self._precision_flag == 16:
                    # this is not deprecated to ease transition between accelerator environments
                    rank_zero_warn(
                        f"You passed `Trainer(accelerator='tpu', precision=16)` but {self._amp_type_flag.value} AMP"
                        f" is not supported with TPUs. Using `precision='bf16'` instead."
                    )
                self.precision_plugin = TPUBf16PrecisionPlugin()
        if self._strategy_flag == "deepspeed" or isinstance(self._strategy_flag, DeepSpeedStrategy):
            self.precision_plugin = DeepSpeedPrecisionPlugin(self._precision_flag, self._amp_type_flag, self._amp_level_flag)

        if self._precision_flag == 32:
            self.precision_plugin = PrecisionPlugin()
        if self._precision_flag == 64:
            self.precision_plugin = DoublePrecisionPlugin()

        # maybe convert the precision value
        if self._precision_flag == 16 and self._accelerator_flag == "cpu":
            # this automatic switch is to ease transition between accelerator environments
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
                device = "cpu" if self._accelerator_flag=="cpu" else "cuda"

                # TODO in progress implement the two following shard types
                # if self._is_sharded_training_type:
                #     return ShardedNativeMixedPrecisionPlugin(self._precision_flag, device)
                # if self._is_fully_sharded_training_type:
                #     return FullyShardedNativeMixedPrecisionPlugin(self._precision_flag, device)
                # return NativeMixedPrecisionPlugin(self._precision_flag, device)


                self._amp_level_flag = self._amp_level_flag or "O2"
                self.precision_plugin = ApexMixedPrecisionPlugin(self._amp_level_flag)
        self.precision_plugin = PrecisionPlugin()

    def _precision_misconfig_check(self):
        if self._accelerator_flag == "ipu":
            if self._precision_flag not in (16, 32):
                raise MisconfigurationException(
                    f"`Trainer(accelerator='ipu', precision={self._precision_flag!r})` is not supported."
                )
        if self._accelerator_flag == "tpu" and self._precision_flag == 64:
                raise MisconfigurationException(
                    "`Trainer(accelerator='tpu', precision=64)` is not implemented."
                    " Please, open an issue in `https://github.com/PyTorchLightning/pytorch-lightning/issues`"
                    " requesting this feature."
                )
        if self._precision_flag == 16 and self._accelerator_flag == "cpu" and self._amp_type_flag == AMPType.APEX:
                # apex was explicitly passed, not a good idea to silently switch to native AMP
                raise MisconfigurationException(
                    "You passed `Trainer(accelerator='cpu', precision=16, amp_type='apex')`"
                    " but apex AMP not supported on CPU."
                )
        if self._precision_flag == "bf16" and self._amp_type_flag != AMPType.NATIVE:
            raise MisconfigurationException(
                f"You passed `Trainer(amp_type={self._amp_type_flag.value!r}, precision='bf16')` but it's not supported."
                " Try using `amp_type='native'` instead."
            )

        # if self._precision_flag in (16, "bf16") and self._amp_type_flag == AMPType.APEX:
        #     if self._is_sharded_training_type or self._is_fully_sharded_training_type:
        #         raise MisconfigurationException(
        #             "Sharded plugins are not supported with apex, please switch to `amp_backend='native'`."
        #         )


    def _init_strategy(self):
        if isinstance(self._strategy_flag, str):
            self.strategy = StrategyRegistry.get(self._strategy_flag)
        else:
            self.strategy = self._strategy_flag
        self.strategy.accelerator = self.accelerator
        if self.precision_plugin:
            self.strategy.precision_plugin = self.precision_plugin
        if self.checkpoint_io:
            self.strategy.checkpoint_io = self.checkpoint_io
        self.strategy.cluster_environment = self.cluster_environment





    ##############################################################################
    # the following logic should be deprecated/removed
    # Added here to keep backward compabilities

    # @property
    # def parallel_devices(self) -> List[Union[torch.device, int]]:
    #     return self._parallel_device

    # @property
    # def replace_sampler_ddp():
    #     return self.replace_sampler_ddp

    # def _distrib_type():

    # def _device_type():

    # def num_nodes():

    # def num_processes():

    # def root_gpu():

    def devices(self):
        return len(self._parallel_devices)

    # def parallel_device_ids():

    # def gpus():

    # def is_distributed():

    def has_ipu(self):
        return self._accelerator_flag == "ipu"

    def has_tpu(self):
        return self._accelerator_flag == "tpu"
