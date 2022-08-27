import ipaddress
import logging
import os
import platform
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import Tensor

import pytorch_lightning as pl
from pytorch_lightning.strategies.strategy import Strategy, TBroadcast
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.data import extract_batch_size
from pytorch_lightning.utilities.enums import PrecisionType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _HIVEMIND_AVAILABLE
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.types import _LRScheduler, ReduceLROnPlateau

if _HIVEMIND_AVAILABLE:
    import hivemind
else:
    hivemind = None

log = logging.getLogger(__name__)


class HivemindStrategy(Strategy):
    INITIAL_PEERS_ENV: str = "PL_INITIAL_PEERS"

    def __init__(
        self,
        target_batch_size: int,
        run_id: str = "lightning_run",
        batch_size: Optional[int] = None,
        delay_state_averaging: bool = False,
        delay_optimizer_step: Optional[bool] = None,
        delay_grad_averaging: bool = False,
        offload_optimizer: Optional[bool] = None,
        reuse_grad_buffers: bool = False,
        scheduler_fn: Optional[Callable] = None,
        matchmaking_time: float = 5.0,
        averaging_timeout: float = 30.0,
        verbose: bool = False,
        averager_opts: Optional[Dict] = None,
        host_maddrs: Optional[List] = None,
        initial_peers: Optional[Union[str, List]] = None,
        **optimizer_kwargs: Any,
    ):
        """Provides capabilities to train using the Hivemind Library, training collaboratively across the internet
        with unreliable machines. For more information, `refer to the docs <https://pytorch-
        lightning.readthedocs.io/en/latest/strategies/hivemind.html>`__.

        .. warning:: ``HivemindStrategy`` is experimental and subject to change.

        Arguments:

            target_batch_size: When training, the batch size to accumulate to before running a step. The larger this
                batch size, the more work can be done asynchronously without communication.

            run_id: A unique identifier of this training run, used as a common prefix for all DHT keys.
                See ``https://learning-at-home.readthedocs.io/en/latest/user/dht.html``.

            batch_size: The local batch size per process. If not provided, we infer this from the first batch of data
                passed in at training (lazy). Note that this should not change throughout training.

            delay_state_averaging: If enabled (default), average parameters and extra tensors in a background thread;
                if set to False, average parameters synchronously within the
                corresponding :meth:`hivemind.Optimizer.step` call.

            delay_optimizer_step: Run optimizer in background, apply results in future .step. requires
                :paramref:`~pytorch_lightning.strategies.hivemind.HivemindStrategy.offload_optimizer`.

            delay_grad_averaging: Average gradients in background; requires
                :paramref:`~pytorch_lightning.strategies.hivemind.HivemindStrategy.offload_optimizer` and
                :paramref:`~pytorch_lightning.strategies.hivemind.HivemindStrategy.delay_optimizer_step`.

            offload_optimizer: Offload the optimizer to host memory, saving GPU memory for parameters and gradients.

            reuse_grad_buffers: Use the model's gradient buffers (params.grad) for gradient accumulation
                which is more memory efficient. Lightning will automatically disable ``zero_grad``
                in the ``LightningModule``.

            scheduler_fn: callable(optimizer) -> PyTorch LRScheduler or a pre-initialized PyTorch scheduler.
                When using `offload_optimizer`/`delay_optimizer_step`/`delay_state_averaging` ``scheduler_fn``
                is required to be passed to the ``HivemindStrategy``. This is because the optimizer
                is re-created and the scheduler needs to be re-created as well.

            matchmaking_time: When looking for group, wait for peers to join for up to this many seconds.
                Increase if you see "averaged gradients with N peers" where N is below 0.9x on >=25% of epochs.
                Training with low-latency network, decreasing matchmaking_time allows training with smaller batch sizes.

            averaging_timeout: If an averaging step hangs for this long, it will be cancelled automatically.
                Increase averaging_timeout if you see "Proceeding with local gradients" at least 25% of the time.
                Do not set this timeout too high, as it may cause your optimizer to hang
                after some types of network errors.

            verbose: Report internal Hivemind events such as accumulating gradients and running background tasks.

            averager_opts: Additional keyword arguments forwarded to both
                ``GradientAverager`` and ``TrainingStateAverager``.

            host_maddrs: List of multi-addrs to create visible peers for other processes.
                `https://learning-at-home.readthedocs.io/en/latest/user/dht.html#running-across-the-internet`

            initial_peers: If connecting to a running process, a list of initial peers needs to be passed in.
                This can also be set via the env variable ``INITIAL_PEERS``.

            **optimizer_kwargs: kwargs are passed to the :class:`hivemind.Optimizer` class.
        """
        if not _HIVEMIND_AVAILABLE or platform.system() != "Linux":
            raise MisconfigurationException(
                "To use the `HivemindStrategy`, you must have Hivemind installed and be running on Linux."
                " Install it by running `pip install -U hivemind`."
            )

        super().__init__()
        self._initial_peers = initial_peers
        self._target_batch_size = target_batch_size
        self._batch_size = batch_size
        self._scheduler_fn = scheduler_fn
        self._require_scheduler_fn = delay_optimizer_step or delay_state_averaging or offload_optimizer
        self._opt = None
        self._optimizer_zero_grad_original: Optional[Callable] = None
        self._run_id = run_id
        self._reuse_grad_buffers = reuse_grad_buffers
        self._optimizer_kwargs = dict(
            matchmaking_time=matchmaking_time,
            averaging_timeout=averaging_timeout,
            delay_optimizer_step=delay_optimizer_step,
            delay_state_averaging=delay_state_averaging,
            delay_grad_averaging=delay_grad_averaging,
            offload_optimizer=offload_optimizer,
            averager_opts=averager_opts if averaging_timeout is not None else dict(request_timeout=1.0),
            verbose=verbose,
            reuse_grad_buffers=reuse_grad_buffers,
            **optimizer_kwargs,
        )

        self._parse_env_initial_peers()

        self.dht = hivemind.DHT(
            start=True,
            initial_peers=initial_peers,
            host_maddrs=host_maddrs if host_maddrs is not None else ["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
        )

        visible_addresses = [
            str(a) for a in self.dht.get_visible_maddrs() if not ipaddress.ip_address(a.values()[0]).is_loopback
        ]

        if initial_peers is None:
            log.info(
                "\nOther machines can connect running the same command:\n"
                f"INITIAL_PEERS={','.join(visible_addresses)} python ...\n"
                "or passing the peers to the strategy:\n"
                f"HivemindStrategy(initial_peers='{','.join(visible_addresses)}')"
            )

        self._hivemind_initialized = False

    def _parse_env_initial_peers(self) -> None:
        initial_peers = os.environ.get(self.INITIAL_PEERS_ENV, self._initial_peers)
        self._initial_peers = initial_peers.split(",") if isinstance(initial_peers, str) else self._initial_peers

    @property
    def num_peers(self) -> int:
        if self._opt:
            return self._opt.tracker.global_progress.num_peers
        return 1

    @property
    def root_device(self) -> torch.device:
        from pytorch_lightning.accelerators.cpu import CPUAccelerator
        from pytorch_lightning.accelerators.cuda import CUDAAccelerator

        if isinstance(self.accelerator, CUDAAccelerator):
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        elif isinstance(self.accelerator, CPUAccelerator):
            return torch.device("cpu")
        raise MisconfigurationException(
            f"Was unable to infer device type from the accelerator: {self.accelerator.__class__.__name__}."
        )

    @property
    def global_rank(self) -> int:
        return 0

    @property
    def is_global_zero(self) -> bool:
        return True

    def setup(self, trainer: "pl.Trainer") -> None:
        self.model_to_device()
        super().setup(trainer)
        if self.precision_plugin.precision in (PrecisionType.HALF, PrecisionType.MIXED):
            self.precision_plugin.scaler = hivemind.GradScaler()

    def _initialize_hivemind(self) -> None:
        if len(self.optimizers) > 1:
            raise MisconfigurationException("Hivemind only supports training with one optimizer.")
        optimizer = self.optimizers[0]

        if self._require_scheduler_fn and self._scheduler_fn is None:
            rank_zero_warn(
                "Enabling `delay_optimizer_step`, `delay_state_averaging` or `offload_optimizer` "
                "requires a `scheduler_fn` to be passed to the strategy if a scheduler is being used "
                "(this is because the optimizer is re-created within Hivemind)."
            )

        scheduler = self._scheduler_fn if self._require_scheduler_fn else None
        params = optimizer.param_groups if self._require_scheduler_fn else None
        optimizer = type(optimizer) if self._require_scheduler_fn else optimizer
        opt = hivemind.Optimizer(
            dht=self.dht,
            run_id=self._run_id,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            target_batch_size=self._target_batch_size,
            batch_size_per_step=self._batch_size,
            **self._optimizer_kwargs,
        )

        if not self._scheduler_fn:
            self._wrap_schedulers(opt)
        opt.load_state_from_peers()
        self.optimizers = [opt]
        self._opt = opt

        if self._reuse_grad_buffers:
            assert self.lightning_module is not None
            self._optimizer_zero_grad_original = self.lightning_module.optimizer_zero_grad
            self._disable_zero_grad()

    def _disable_zero_grad(self) -> None:
        lightning_module = self.lightning_module
        if is_overridden("optimizer_zero_grad", lightning_module):
            assert lightning_module is not None  # `is_overridden` returns False otherwise
            rank_zero_warn(
                "You have overridden `optimizer_zero_grad` which will be disabled."
                " When `HivemindStrategy(reuse_grad_buffers=True)`, the optimizer cannot call zero grad,"
                " as this would delete the gradients before they are averaged."
            )
        assert lightning_module is not None
        lightning_module.optimizer_zero_grad = None  # type: ignore[assignment]

    def _wrap_schedulers(self, opt: "hivemind.Optimizer") -> None:
        # wrap schedulers so that they only update when the hivemind optimizer updates
        for scheduler_config in self.lr_scheduler_configs:
            scheduler = scheduler_config.scheduler
            if isinstance(scheduler, ReduceLROnPlateau):
                raise ValueError(
                    f"The `ReduceLROnPlateau` scheduler is not currently supported with `{self.__class__.__name__}`."
                )
            scheduler_config.scheduler = HiveMindScheduler(
                optimizer=opt,
                scheduler=scheduler,
            )

    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if not self._hivemind_initialized:
            self._hivemind_initialized = True
            # todo (sean): we could technically support a dynamic batch size by inferring each step
            # and passing it to the ``hivemind.Optimizer``.
            if self._batch_size is None:
                try:
                    self._batch_size = extract_batch_size(batch)
                    log.info(f"Found per machine batch size automatically from the batch: {self._batch_size}")
                except (MisconfigurationException, RecursionError) as e:
                    raise MisconfigurationException(
                        "We tried to infer the batch size from the first batch of data. "
                        "Please provide the batch size to the Strategy by "
                        "``Trainer(strategy=HivemindStrategy(batch_size=x))``. "
                    ) from e
            self._initialize_hivemind()

    def reduce(self, tensor: Union[Any, Tensor], *args: Any, **kwargs: Any) -> Union[Any, Tensor]:
        return tensor

    def all_gather(self, tensor: Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> Tensor:
        return tensor

    def model_to_device(self) -> None:
        assert self.model is not None
        self.model.to(self.root_device)

    def barrier(self, *args: Any, **kwargs: Any) -> None:
        pass

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        return obj

    def teardown(self) -> None:

        if self._optimizer_zero_grad_original is not None and self.lightning_module is not None:
            # re-enable `optimizer_zero_grad`
            self.lightning_module.optimizer_zero_grad = self._optimizer_zero_grad_original  # type: ignore[assignment]

        if self._opt:
            self._opt.shutdown()
        log.info("Shutting down hivemind DHT.")
        self.dht.shutdown()
        super().teardown()


class HiveMindScheduler:
    """Wrapper for schedulers to prevent Lightning from stepping the scheduler too soon.

    This code ensures that we only step when the HiveMind optimizer reaches the global step.
    """

    base_lrs: List[float]

    def __init__(self, optimizer: "hivemind.Optimizer", scheduler: _LRScheduler) -> None:
        # copy most of the `Scheduler` methods into this instance. `__del__` is skipped in case the scheduler has
        # implemented custom logic which we would not want to call on destruction of the `HiveMindScheduler`
        self.__dict__ = {k: v for k, v in scheduler.__dict__.items() if k not in ("step", "__del__")}

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.current_step = -1

    def step(self, epoch: Optional[int] = None) -> None:
        while self.current_step < self.optimizer.local_epoch:
            self.scheduler.step(epoch=epoch)
            self.current_step += 1

    def load_state_dict(self, state_dict: Dict) -> None:
        self.scheduler.load_state_dict(state_dict)

    def state_dict(self) -> Dict:
        return self.scheduler.state_dict()
