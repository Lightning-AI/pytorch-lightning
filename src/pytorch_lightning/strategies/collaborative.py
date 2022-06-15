import http
import ipaddress
import logging
import os
import platform
import re
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler
from typing import Any, Callable, Dict, List, Optional, Union

import requests
import torch
from torch import Tensor

import pytorch_lightning as pl
from pytorch_lightning.strategies.strategy import Strategy, TBroadcast
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
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


class CollaborativeStrategy(Strategy):
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
        endpoint: Optional[bool] = None,
        peer_endpoint: Optional[str] = None,
        persistent: bool = True,
        host: Optional[str] = None,
        port: Optional[int] = None,
        retry_endpoint_attempts: int = 5,
        retry_endpoint_sleep_duration: int = 5,
        **optimizer_kwargs: Any,
    ):
        """Provides capabilities to train using the Hivemind Library, training collaboratively across the internet
        with unreliable machines. For more information, `refer to the docs <https://pytorch-
        lightning.readthedocs.io/en/latest/strategies/collaborative_training.html>`__.

        .. warning:: ``CollaborativeStrategy`` is experimental and subject to change.

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
                :paramref:`~pytorch_lightning.strategies.collaborative.CollaborativeStrategy.offload_optimizer`.

            delay_grad_averaging: Average gradients in background; requires
                :paramref:`~pytorch_lightning.strategies.collaborative.CollaborativeStrategy.offload_optimizer` and
                :paramref:`~pytorch_lightning.strategies.collaborative.CollaborativeStrategy.delay_optimizer_step`.

            offload_optimizer: Offload the optimizer to host memory, saving GPU memory for parameters and gradients.

            reuse_grad_buffers: Use the model's gradient buffers (params.grad) for gradient accumulation
                which is more memory efficient. Lightning will automatically disable ``zero_grad``
                in the ``LightningModule``.

            scheduler_fn: callable(optimizer) -> PyTorch LRScheduler or a pre-initialized PyTorch scheduler.
                When using `offload_optimizer`/`delay_optimizer_step`/`delay_state_averaging` ``scheduler_fn``
                is required to be passed to the ``CollaborativeStrategy``. This is because the optimizer
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

            endpoint: Enable if a side-car endpoint server is required on the process to server initial peers.
                This is useful when using some form of orchestration such as torchelastic.

            peer_endpoint: The endpoint to request initial peers from.

            persistent: When using an endpoint, this controls whether other processes that are not the endpoint
                server log/checkpoint. If ``persistent`` is True, we do not log/checkpoint from other processes.

            host: When creating the endpoint, the host IP to use.

            port: When creating the endpoint, the host port to use.

            retry_endpoint_attempts: When connecting to the
                :paramref:`~pytorch_lightning.strategies.collaborative.CollaborativeStrategy.peer_endpoint`,
                how many time to retry before raising an exception.

            retry_endpoint_sleep_duration: When connecting to the
                :paramref:`~pytorch_lightning.strategies.collaborative.CollaborativeStrategy.peer_endpoint`,
                how long to wait between retries.

            **optimizer_kwargs: kwargs are passed to the :class:`hivemind.Optimizer` class.
        """
        if not _HIVEMIND_AVAILABLE or platform.system() != "Linux":
            raise MisconfigurationException(
                "To use the `CollaborativeStrategy`, you must have Hivemind installed and be running on Linux."
                " Install it by running `pip install -U hivemind`."
            )

        super().__init__()
        self.dht_manager = DHTManager(
            persistent=persistent,
            endpoint=endpoint,
            peer_endpoint=peer_endpoint,
            host=host,
            port=port,
            host_maddrs=host_maddrs,
            initial_peers=initial_peers,
            retry_endpoint_attempts=retry_endpoint_attempts,
            retry_endpoint_sleep_duration=retry_endpoint_sleep_duration,
        )
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

        # a bit of a hack to only log from the stable server
        if self.dht_manager.disable_logging_checkpointing:
            warnings.warn(
                "This machine is not a persistent machine. Checkpointing/Logging has been disabled.", UserWarning
            )
        rank_zero_only.rank = 1 if self.dht_manager.disable_logging_checkpointing else 0
        self._hivemind_initialized = False

    @property
    def num_peers(self) -> int:
        if self._opt:
            return self._opt.tracker.global_progress.num_peers
        return 1

    @property
    def dht(self) -> "hivemind.DHT":
        """Hivemind Distributed Hash Table which stores values across all peers.

        See documentation for more details: `https://learning-at-home.readthedocs.io/en/latest/modules/dht.html`
        """
        return self.dht_manager.dht

    @property
    def root_device(self) -> torch.device:
        from pytorch_lightning.accelerators.cpu import CPUAccelerator
        from pytorch_lightning.accelerators.gpu import GPUAccelerator

        if isinstance(self.accelerator, GPUAccelerator):
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
                " When `CollaborativeStrategy(reuse_grad_buffers=True)`, the optimizer cannot call zero grad,"
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
                        "``Trainer(strategy=CollaborativeStrategy(batch_size=x))``. "
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


class DHTManager:
    ENDPOINT_ENV: str = "PL_ENDPOINT"
    PEER_ENDPOINT_ENV: str = "PL_PEER_ENDPOINT"
    INITIAL_PEERS_ENV: str = "PL_INITIAL_PEERS"
    HOST_ENV: str = "PL_HOST"
    PORT_ENV: str = "PL_PORT"
    DEFAULT_HOST: str = "0.0.0.0"
    DEFAULT_PORT: int = 1440

    def __init__(
        self,
        host_maddrs: Optional[List],
        initial_peers: Optional[Union[str, List]],
        persistent: bool,
        endpoint: Optional[bool],
        peer_endpoint: Optional[str],
        host: Optional[str],
        port: Optional[int],
        retry_endpoint_attempts: int = 5,
        retry_endpoint_sleep_duration: int = 5,
    ) -> None:
        """Manages the `hivemind.DHT` connection and provides a side-car endpoint server for initial peer access.

        Arguments:

            host_maddrs: :paramref:`~pytorch_lightning.strategies.collaborative.CollaborativeStrategy.host_maddrs`

            initial_peers: :paramref:`~pytorch_lightning.strategies.collaborative.CollaborativeStrategy.initial_peers`

            persistent: :paramref:`~pytorch_lightning.strategies.collaborative.CollaborativeStrategy.persistent`

            endpoint: :paramref:`~pytorch_lightning.strategies.collaborative.CollaborativeStrategy.endpoint`

            peer_endpoint: :paramref:`~pytorch_lightning.strategies.collaborative.CollaborativeStrategy.peer_endpoint`

            host: :paramref:`~pytorch_lightning.strategies.collaborative.CollaborativeStrategy.host`

            port: :paramref:`~pytorch_lightning.strategies.collaborative.CollaborativeStrategy.port`

            retry_endpoint_attempts:
                :paramref:`~pytorch_lightning.strategies.collaborative.CollaborativeStrategy.retry_endpoint_attempts`

            retry_endpoint_sleep_duration:
                :paramref:
                `~pytorch_lightning.strategies.collaborative.CollaborativeStrategy.retry_endpoint_sleep_duration`
        """
        self._persistent = persistent
        self._endpoint = endpoint
        self._initial_peers = initial_peers
        self._peer_endpoint = peer_endpoint
        self._host = host
        self._port = port

        self._parse_env_vars()

        if self._peer_endpoint and self._initial_peers is None:
            self._initial_peers = self._get_initial_peers_from_endpoint(
                retry_initial_peers=retry_endpoint_attempts, retry_peer_sleep_duration=retry_endpoint_sleep_duration
            )

        self.dht = hivemind.DHT(
            start=True,
            initial_peers=self._initial_peers,
            host_maddrs=host_maddrs if host_maddrs is not None else ["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
        )

        visible_addresses = [
            str(a) for a in self.dht.get_visible_maddrs() if not ipaddress.ip_address(a.values()[0]).is_loopback
        ]

        if self._endpoint:
            self._host = self._host if self._host is not None else self.DEFAULT_HOST
            self._port = self._port if self._port is not None else self.DEFAULT_PORT
            self._start_server_process(self._host, self._port)
            self._log_endpoint_helper_message(visible_addresses)
        elif self._peer_endpoint:
            log.info("Machine received initial peers from endpoint.")
        elif self._initial_peers is None:
            log.info(
                "\nOther machines can connect running the same command:\n"
                f"INITIAL_PEERS={','.join(visible_addresses)} python ...\n"
                "or passing the peers to the strategy:\n"
                f"CollaborativeStrategy(initial_peers='{','.join(visible_addresses)}')"
            )

    def _log_endpoint_helper_message(self, visible_addresses: List[str]) -> None:
        assert self._host is not None
        resolved_host = self._host
        if "0.0.0.0" in self._host:
            # use the visible multi-addresses to figure out the IP that has been exposed
            # todo (sean): this is pretty hacky, worth investigating.
            p = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+")
            # todo (sean): we select one address from here, could we have multiple?
            resolved_host = {p.findall(maddr)[0] for maddr in visible_addresses}.pop()
        log.info(
            "\nSidecar endpoint enabled to serve peers.\n"
            "Other peers can connect via:\n"
            f"PEER_ENDPOINT={resolved_host}:{self._port} python ...\n"
            "or pass the peer endpoint address to the strategy:\n"
            f"CollaborativeStrategy(peer_endpoint='{resolved_host}:{self._port}')"
        )

    def _start_server_process(self, host: str, port: int) -> None:
        dht = self.dht

        class DHTHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                """Respond to a GET request."""
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()

                visible_peers = [
                    str(a) for a in dht.get_visible_maddrs() if not ipaddress.ip_address(a.values()[0]).is_loopback
                ]

                self.wfile.write("\n".join(visible_peers).encode())

        server = http.server.ThreadingHTTPServer((host, int(port)), DHTHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()

    def _get_initial_peers_from_endpoint(self, retry_initial_peers: int, retry_peer_sleep_duration: int) -> List:
        peers = None
        for _ in range(retry_initial_peers):
            try:
                peers = self._get_peers()
                break
            except requests.exceptions.RequestException:
                log.info(f"Failed to get peers, retrying in {retry_peer_sleep_duration} seconds...")
                time.sleep(retry_peer_sleep_duration)
        if peers is None:
            raise MisconfigurationException(
                f"Unable to get peers. Tried {retry_initial_peers} times waiting {retry_peer_sleep_duration}s."
                f"These parameters can be extended by passing "
                "to the strategy (CollaborativeStrategy(retry_connection=x, retry_sleep_duration=y))."
            )
        log.info(f"Received initial peers from collaborative server: {peers}")
        return peers

    def _get_peers(self) -> List[str]:
        assert self._peer_endpoint is not None
        url = f"http://{self._peer_endpoint}" if not self._peer_endpoint.startswith("http://") else self._peer_endpoint
        r = requests.get(url)
        return r.text.split(",")

    def _parse_env_vars(self) -> None:
        endpoint = os.environ.get(self.ENDPOINT_ENV, self._endpoint)
        self._endpoint = endpoint == "1" if isinstance(endpoint, str) else endpoint
        self._peer_endpoint = os.environ.get(self.PEER_ENDPOINT_ENV, self._peer_endpoint)
        initial_peers = os.environ.get(self.INITIAL_PEERS_ENV, self._initial_peers)
        self._initial_peers = initial_peers.split(",") if isinstance(initial_peers, str) else initial_peers

        port = os.environ.get(self.PORT_ENV, self._port)
        self._port = int(port) if isinstance(port, str) else port
        self._host = os.environ.get(self.HOST_ENV, self._host)

    @property
    def disable_logging_checkpointing(self) -> bool:
        # if this node is a peer, we do not log/checkpoint in persistent mode.
        return self._persistent and (self._initial_peers is not None or self._peer_endpoint is not None)
