import http
import ipaddress
import logging
import os
import re
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler
from typing import Any, Callable, Dict, List, Optional, Union

import requests
import torch

import pytorch_lightning as pl
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.utilities import _XLA_AVAILABLE, rank_zero_only, rank_zero_warn
from pytorch_lightning.utilities.data import extract_batch_size
from pytorch_lightning.utilities.enums import PrecisionType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _HIVEMIND_AVAILABLE
from pytorch_lightning.utilities.model_helpers import is_overridden

if _HIVEMIND_AVAILABLE:
    import hivemind

log = logging.getLogger(__name__)


class CollaborativeStrategy(Strategy):
    def __init__(
        self,
        target_batch_size: int,
        run_id: str = "lightning_run",
        batch_size: Optional[int] = None,
        delay_state_averaging: bool = False,
        delay_optimizer_step: bool = False,
        delay_grad_averaging: bool = False,
        offload_optimizer: bool = False,
        reuse_grad_buffers: bool = False,
        scheduler_fn: Optional[Callable] = None,
        matchmaking_time: float = 5.0,
        averaging_timeout: float = 30.0,
        grad_compression: hivemind.CompressionBase = hivemind.Float16Compression(),
        state_averaging_compression: hivemind.CompressionBase = hivemind.Float16Compression(),
        verbose: bool = True,
        averager_opts: Optional[Dict] = None,
        host_maddrs: Optional[List] = None,
        initial_peers: Optional[List] = None,
        endpoint: Optional[bool] = None,
        peer_endpoint: Optional[str] = None,
        persistent: bool = True,
        host: Optional[str] = None,
        port: Optional[int] = None,
        retry_initial_peers: int = 5,
        retry_peer_sleep_duration: int = 5,
        **optimizer_kwargs,
    ):
        """Provides capabilities to train using the Hivemind Library, training collaboratively across the internet
        with unreliable machines. `For more information: https://pytorch-
        lightning.readthedocs.io/en/latest/advanced/collaborative_training.html`.

        Arguments:

            target_batch_size: When training, the batch size to accumulate to before running a step. The larger this
            batch size, the more work can be done asynchronously without communication.

            run_id: A unique identifier of this training run, used as a common prefix for all DHT keys.
            batch_size: The local batch size per process. If not provided, we infer this from the first batch of data
            passed in at training.

            delay_state_averaging: If enabled (default), average parameters and extra tensors in a background thread;
            if set to False, average parameters synchronously within the corresponding hivemind.Optimizer.step call.

            delay_optimizer_step: Run optimizer in background, apply results in future .step.
            requires `offload_optimizer`.

            delay_grad_averaging: Average gradients in background; requires offload_optimizer and delay_optimizer_step.

            offload_optimizer: offload the optimizer to host memory, saving GPU memory for parameters and gradients.

            reuse_grad_buffers: Use model's .grad buffers for gradient accumulation.
              This is more memory efficient, but it requires that we do not call `zero_grad` in the `LightningModule`.

            scheduler_fn: callable(optimizer) -> PyTorch LRScheduler or a pre-initialized PyTorch scheduler.
            When using `offload_optimizer`/`delay_optimizer_step`/`delay_state_averaging` a scheduler_fn is required
            instead of passing a scheduler to the `pl.LightningModule`. This is because the optimizer is re-created
            and the scheduler needs to be re-created as well.

            matchmaking_time: When looking for group, wait for peers to join for up to this many seconds.
            Increase if you see "averaged gradients with N peers" where N is below 0.9x on >=25% of epochs.
            Training with low-latency network, decreasing matchmaking_time allows training with smaller batch sizes.

            averaging_timeout: If an averaging step hangs for this long, it will be cancelled automatically.
            Increase averaging_timeout if you see "Proceeding with local gradients" at least 25% of the time.
            Do not set this timeout too high, as it may cause your optimizer to hang after some types of network errors.

            allreduce_timeout: Timeout for a single attempt to run all-reduce.

            grad_compression: Compression strategy used for averaging gradients.

            state_averaging_compression: Compression for averaging params and state tensors.

            verbose: Report internal Hivemind events such as accumulating gradients and running background tasks.

            averager_opts: Additional keyword arguments forwarded to both GradientAverager and TrainingStateAverager.

            host_maddrs: List of multi-addrs to create visible peers for other processes.
            https://learning-at-home.readthedocs.io/en/latest/user/dht.html#running-across-the-internet

            initial_peers: If connecting to a running process, a list of initial peers needs to be passed in.
            This can also be set via the env variable `INITIAL_PEERS`.

            endpoint: Enable if a side-car endpoint server is required on the process to server initial peers.
            This is useful when using some form of orchestration such as torchelastic.

            peer_endpoint: The endpoint to request initial peers from.

            persistent: When using an endpoint, this controls whether other processes that are not the endpoint
            server log/checkpoint. If `persistent` is True, we do not log/checkpoint from other processes.

            host: When creating the endpoint, the host IP to use.

            port: When creating the endpoint, the host port to use.

            retry_initial_peers: When connecting to the `peer_endpoint`, how many time to retry before raising
            an exception.

            retry_peer_sleep_duration: When connecting to the `peer_endpoint`, how long to wait between retries.

            optimizer_kwargs: kwargs are passed to the `hivemind.Optimizer` class.
        """
        super().__init__()
        self.dht_manager = DHTManager(
            persistent=persistent,
            endpoint=endpoint,
            peer_endpoint=peer_endpoint,
            host=host,
            port=port,
            host_maddrs=host_maddrs,
            initial_peers=initial_peers,
            retry_initial_peers=retry_initial_peers,
            retry_peer_sleep_duration=retry_peer_sleep_duration,
        )
        self.target_batch_size = target_batch_size
        self.batch_size = batch_size
        self.scheduler_fn = scheduler_fn
        self.delay_state_averaging = delay_state_averaging
        self.delay_optimizer_step = delay_optimizer_step
        self.offload_optimizer = offload_optimizer
        self.opt = None
        self.run_id = run_id
        self.reuse_grad_buffers = reuse_grad_buffers
        self.optimizer_kwargs = dict(
            matchmaking_time=matchmaking_time,
            averaging_timeout=averaging_timeout,
            delay_optimizer_step=delay_optimizer_step,
            delay_state_averaging=delay_state_averaging,
            delay_grad_averaging=delay_grad_averaging,
            offload_optimizer=offload_optimizer,
            grad_compression=grad_compression,
            state_averaging_compression=state_averaging_compression,
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
        self.hivemind_initialized = False

        self.global_rank = 0
        self.local_rank = 0
        self.world_size = 1

    def _initialize_hivemind(self, trainer: "pl.Trainer") -> None:
        if len(trainer.optimizers) > 1:
            raise MisconfigurationException("Hivemind only supports training with one optimizer.")
        (optimizer,) = trainer.optimizers

        enabling_features = self.delay_optimizer_step or self.delay_state_averaging or self.offload_optimizer

        if enabling_features and self.scheduler_fn is None:
            rank_zero_warn(
                "Enabling delay_optimizer_step, delay_state_averaging or offload_optimizer "
                "requires a scheduler_fn to be passed to the strategy if a scheduler is being used "
                "(this is because the optimizer is re-created within Hivemind)."
            )

        scheduler = self.scheduler_fn if enabling_features else None
        params = optimizer.param_groups if enabling_features else None
        optimizer = type(optimizer) if enabling_features else optimizer
        opt = hivemind.Optimizer(
            dht=self.dht,
            run_id=self.run_id,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            target_batch_size=self.target_batch_size,
            batch_size_per_step=self.batch_size,
            **self.optimizer_kwargs,
        )

        if not self.scheduler_fn:
            self._wrap_schedulers(opt, trainer)
        opt.load_state_from_peers()
        trainer.optimizers = [opt]
        self.opt = opt

        if self.reuse_grad_buffers:
            # turn off zero_grad by Lightning
            if is_overridden("optimizer_zero_grad", self.lightning_module):
                raise rank_zero_warn(
                    "You have overridden `optimizer_zero_grad` which will be dangerous. "
                    "When CollaborativeStrategy(reuse_grad_buffers=True), "
                    "the optimizer cannot call zero grad, as this would "
                    "delete the gradients before they are averaged."
                )

            def override_fn(*args, **kwargs):
                pass

            self.lightning_module.optimizer_zero_grad = override_fn

    def _wrap_schedulers(self, opt: "hivemind.Optimizer", trainer: "pl.Trainer") -> None:
        # wrap schedulers so that they only update when the hivemind optimizer updates
        for scheduler_config in trainer.lr_scheduler_configs:
            scheduler_config.scheduler = HiveMindScheduler(
                scheduler=scheduler_config.scheduler,
                optimizer=opt,
            )

    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if not self.hivemind_initialized:
            self.hivemind_initialized = True
            if self.batch_size is None:
                try:
                    self.batch_size = extract_batch_size(batch)
                    log.info(f"Found per machine batch size automatically from the batch: {self.batch_size}")
                except Exception as e:
                    raise MisconfigurationException(
                        "We tried to infer the batch size from the first batch of data are were unable to. "
                        "Please provide the batch size to the Strategy (CollaborativeStrategy"
                        f"exception raised: {e}"
                    )
            self._initialize_hivemind(self.lightning_module.trainer)

        self.lightning_module.log("num_peers", self.num_peers)

    @property
    def num_peers(self) -> int:
        return self.opt.tracker.global_progress.num_peers

    @property
    def dht(self) -> "hivemind.DHT":
        return self.dht_manager.dht

    @property
    def on_tpu(self) -> bool:
        return self.root_device.type == "xla" and _XLA_AVAILABLE

    @property
    def on_gpu(self) -> bool:
        return self.root_device.type == "cuda" and torch.cuda.is_available()

    def reduce(self, tensor: Union[Any, torch.Tensor], *args: Any, **kwargs: Any) -> Union[Any, torch.Tensor]:
        return tensor

    def all_gather(self, tensor: torch.Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> torch.Tensor:
        return tensor

    @property
    def root_device(self) -> torch.device:
        # todo: cyclic import, someone show me the way
        from pytorch_lightning.accelerators.cpu import CPUAccelerator
        from pytorch_lightning.accelerators.gpu import GPUAccelerator

        if isinstance(self.accelerator, GPUAccelerator):
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        elif isinstance(self.accelerator, CPUAccelerator):
            return torch.device("cpu")
        raise ValueError

    def model_to_device(self) -> None:
        self.model.to(self.root_device)

    def setup(self, trainer: "pl.Trainer") -> None:
        self.model_to_device()
        super().setup(trainer)
        if self.precision_plugin.precision in (PrecisionType.HALF, PrecisionType.MIXED):
            self.precision_plugin.scaler = hivemind.GradScaler()

    @property
    def is_global_zero(self) -> bool:
        return True

    def barrier(self, *args, **kwargs) -> None:
        pass

    def broadcast(self, obj: object, src: int = 0) -> object:
        return obj

    def teardown(self) -> None:
        super().teardown()
        if self.on_gpu:
            # GPU teardown
            self.lightning_module.cpu()
            # clean up memory
            torch.cuda.empty_cache()
        if self.opt:
            self.opt.shutdown()
        log.info("Shutting down hivemind DHT.")
        self.dht.shutdown()


class HiveMindScheduler:
    """Wrapper for schedulers to prevent Lightning from stepping the scheduler too soon.

    This code ensures that we only step when the HiveMind optimizer reaches the global step.
    """

    def __init__(self, optimizer: hivemind.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler):
        # copy most of the `Scheduler` methods into this instance. `__del__` is skipped in case the scheduler has
        # implemented custom logic which we would not want to call on destruction of the `SwarmyScheduler`
        self.__dict__ = {k: v for k, v in scheduler.__dict__.items() if k not in ("step", "__del__")}

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.current_step = -1

    def step(self, epoch: Optional[int] = None) -> None:
        while self.current_step < self.optimizer.local_epoch:
            self.scheduler.step()
            self.current_step += 1

    def load_state_dict(self, state_dict: Dict):
        self.scheduler.load_state_dict(state_dict)

    def state_dict(self) -> Dict:
        return self.scheduler.state_dict()


class DHTManager:
    ENDPOINT_ENV: str = "ENDPOINT"
    PEER_ENDPOINT_ENV: str = "PEER_ENDPOINT"
    INITIAL_PEERS_ENV: str = "INITIAL_PEERS"
    HOST_ENV: str = "HOST"
    PORT_ENV: str = "PORT"
    DEFAULT_HOST: str = "0.0.0.0"
    DEFAULT_PORT: int = 1440

    def __init__(
        self,
        host_maddrs: Optional[List],
        initial_peers: Optional[List],
        persistent: bool,
        endpoint: Optional[bool],
        peer_endpoint: Optional[str],
        host: Optional[str],
        port: Optional[int],
        retry_initial_peers: int = 5,
        retry_peer_sleep_duration: int = 5,
    ):
        """Manages the `hivemind.DHT` connection and provides a side-car endpoint server for initial peer access.

        Arguments:

            host_maddrs: List of multi-addrs to create visible peers for other processes.
            https://learning-at-home.readthedocs.io/en/latest/user/dht.html#running-across-the-internet

            initial_peers: If connecting to a running process, a list of initial peers needs to be passed in.
            This can also be set via the env variable `INITIAL_PEERS`.

            endpoint: Enable if a side-car endpoint server is required on the process to server initial peers.
            This is useful when using some form of orchestration such as torchelastic.

            peer_endpoint: The endpoint to request initial peers from.

            persistent: When using an endpoint, this controls whether other processes that are not the endpoint
            server log/checkpoint. If `persistent` is True, we do not log/checkpoint from other processes.

            host: When creating the endpoint, the host IP to use.

            port: When creating the endpoint, the host port to use.

            retry_initial_peers: When connecting to the `peer_endpoint`, how many time to retry before raising
            an exception.

            retry_peer_sleep_duration: When connecting to the `peer_endpoint`, how long to wait between retries.
        """
        logging.basicConfig()
        log.setLevel(logging.INFO)
        self.persistent = persistent
        self.endpoint = endpoint
        self.initial_peers = initial_peers
        self.peer_endpoint = peer_endpoint
        self.host = host
        self.port = port

        self._parse_env_vars()

        if self.peer_endpoint and self.initial_peers is None:
            self.initial_peers = self._get_initial_peers_from_endpoint(
                retry_initial_peers=retry_initial_peers, retry_peer_sleep_duration=retry_peer_sleep_duration
            )

        self.dht = hivemind.DHT(
            start=True,
            initial_peers=self.initial_peers,
            host_maddrs=host_maddrs if host_maddrs is not None else ["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
        )

        visible_addresses = [
            str(a) for a in self.dht.get_visible_maddrs() if not ipaddress.ip_address(a.values()[0]).is_loopback
        ]

        if self.endpoint:
            self.host = self.host if self.host is not None else self.DEFAULT_HOST
            self.port = self.port if self.port is not None else self.DEFAULT_PORT
            self._start_server_process(self.host, self.port)
            self._log_endpoint_helper_message(visible_addresses)
        elif self.peer_endpoint:
            log.info("Machine received initial peers from endpoint.")
        elif self.initial_peers is None:
            log.info(
                f"\nOther machines can connect running the same command:\n"
                f"INITIAL_PEERS={','.join(visible_addresses)} python ...\n"
                "or passing the peers to the strategy:\n"
                f"CollaborativeStrategy(initial_peers='{','.join(visible_addresses)}')"
            )

    def _log_endpoint_helper_message(self, visible_addresses: List[str]) -> None:
        resolved_host = self.host
        if "0.0.0.0" in self.host:
            # use the visible multi-addresses to figure out the IP that has been exposed
            # todo: this is pretty hacky, worth investigating.
            p = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+")
            # todo: we select one address from here, could we have multiple?
            resolved_host = {p.findall(maddr)[0] for maddr in visible_addresses}.pop()
        log.info(
            "\nSidecar endpoint enabled to serve peers.\n"
            "Other peers can connect via:\n"
            f"PEER_ENDPOINT={resolved_host}:{self.port} python ...\n"
            "or pass the peer endpoint address to the strategy:\n"
            f"CollaborativeStrategy(peer_endpoint='{resolved_host}:{self.port}')"
        )

    def _start_server_process(self, host: str, port: int) -> None:
        dht = self.dht

        class DHTHandler(BaseHTTPRequestHandler):
            def do_GET(self):
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
        for x in range(retry_initial_peers):
            try:
                peers = self._get_peers()
                break
            except Exception:
                log.info(f"Failed to get peers, retrying in {retry_peer_sleep_duration} seconds...")
                time.sleep(retry_peer_sleep_duration)
        if peers is None:
            raise MisconfigurationException(
                f"Was unable to get peers. Tried {retry_initial_peers} times waiting {retry_peer_sleep_duration}s."
                f"These parameters can be extended by passing "
                "to the strategy (CollaborativeStrategy(retry_connection=x, retry_sleep_duration=y)"
            )
        log.info(f"Received initial peers from collaborative server: {peers}")
        return peers

    def _get_peers(self) -> List[str]:
        url = f"http://{self.peer_endpoint}" if not self.peer_endpoint.startswith("http://") else self.peer_endpoint
        r = requests.get(url)
        return r.text.split(",")

    def _parse_env_vars(self) -> None:
        endpoint = os.environ.get(self.ENDPOINT_ENV, self.endpoint)
        self.endpoint = endpoint == "1" if isinstance(endpoint, str) else endpoint
        self.peer_endpoint = os.environ.get(self.PEER_ENDPOINT_ENV, self.peer_endpoint)
        initial_peers = os.environ.get(self.INITIAL_PEERS_ENV, self.initial_peers)
        self.initial_peers = initial_peers.split(",") if isinstance(initial_peers, str) else initial_peers

        self.port = os.environ.get(self.PORT_ENV, self.port)
        self.host = os.environ.get(self.HOST_ENV, self.host)

    @property
    def disable_logging_checkpointing(self) -> bool:
        if self.persistent and (self.initial_peers or self.peer_endpoint):
            # if this node is a peer, we do not log/checkpoint in persistent mode.
            return True
        return False
