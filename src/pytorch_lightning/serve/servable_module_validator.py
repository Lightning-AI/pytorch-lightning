import time
from multiprocessing import Process
from typing import Any, Dict, Optional

import requests
import torch
from typing_extensions import Literal

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.serve.servable_module import ServableModule
from pytorch_lightning.strategies import DDPFullyShardedNativeStrategy, DDPFullyShardedStrategy, DeepSpeedStrategy
from pytorch_lightning.utilities.imports import _RequirementAvailable
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_only

_NOT_SUPPORTED_STRATEGIES = (
    DeepSpeedStrategy,
    DDPFullyShardedNativeStrategy,
    DDPFullyShardedStrategy,
)


class ServableModuleValidator(Callback):
    """The ServableModuleValidator callback enables to validate a model is servable following the ServableModule
    API.

    Arguments:
        optimization: The format in which the model should be tested while being served.
        server: The library used to evaluate the model serving. Default is FastAPI.
        host: The host address associated the server.
        port: The port associated the server.
    """

    def __init__(
        self,
        optimization: Optional[Literal["trace", "script", "onnx", "tensorRt"]] = None,
        server: Literal["fastapi", "ml_server", "torchserve", "sagemaker"] = "fastapi",
        host: str = "127.0.0.1",
        port: int = 8080,
    ):
        super().__init__()
        fastapi_installed = _RequirementAvailable("fastapi")
        if not fastapi_installed:
            raise ModuleNotFoundError(fastapi_installed.message)
        uvicorn_installed = _RequirementAvailable("uvicorn")
        if not uvicorn_installed:
            raise ModuleNotFoundError(uvicorn_installed.message)

        # TODO: Add support for the other options
        if optimization is not None:
            raise NotImplementedError(f"The optimization {optimization} is currently not supported.")
        # TODO: Add support for testing with those server services
        if server != "fastapi":
            raise NotImplementedError("Only the fastapi server is currently supported.")

        self.optimization = optimization
        self.host = host
        self.port = port
        self.server = server
        self.resp: Optional[requests.Response] = None

    @rank_zero_only
    def on_train_start(self, trainer: "pl.Trainer", servable_module: "pl.LightningModule") -> None:
        if isinstance(trainer.strategy, _NOT_SUPPORTED_STRATEGIES):
            raise Exception(
                f"The current strategy {trainer.strategy} used "
                "by the trainer isn't supported for sanity serving yet."
            )

        if not isinstance(servable_module, ServableModule):
            raise TypeError(f"The provided model should be subclass of {ServableModule.__qualname__}.")

        if not is_overridden("configure_payload", servable_module, ServableModule):
            raise NotImplementedError("The `configure_payload` method needs to be overridden.")
        if not is_overridden("configure_serialization", servable_module, ServableModule):
            raise NotImplementedError("The `configure_serialization` method needs to be overridden.")
        if not is_overridden("serve_step", servable_module, ServableModule):
            raise NotImplementedError("The `serve_step` method needs to be overridden.")

        process = Process(target=self._start_server, args=(servable_module, self.host, self.port, self.optimization))
        process.start()

        ready = False
        while not ready:
            try:
                resp = requests.get(f"http://{self.host}:{self.port}/ping")
                ready = resp.status_code == 200
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(0.1)

        payload = servable_module.configure_payload()

        if "body" not in payload:
            raise Exception(f"Your provided payload {payload} should have a field body.")

        self.resp = requests.post(f"http://{self.host}:{self.port}/serve", json=payload)
        process.kill()

    @property
    def successful(self) -> bool:
        """Returns whether the model was successfully served."""
        return self.resp.status_code == 200 if self.resp else False

    def state_dict(self) -> Dict[str, Any]:
        return {"successful": self.successful, "optimization": self.optimization, "server": self.server}

    @staticmethod
    def _start_server(servable_model: ServableModule, host: str, port: int, _: bool) -> None:
        """This optimization starts a simple FastAPI server with a predict and ping endpoint."""
        from fastapi import Body, FastAPI
        from uvicorn import run

        app = FastAPI()

        deserializers, serializers = servable_model.configure_serialization()
        servable_model.eval()

        @app.get("/ping")
        def ping() -> bool:
            return True

        @app.post("/serve")
        async def serve(payload: dict = Body(...)) -> Dict[str, Any]:
            body = payload["body"]

            for key, deserializer in deserializers.items():
                body[key] = deserializer(body[key])

            with torch.no_grad():
                output = servable_model.serve_step(**body)

            if not isinstance(output, dict):
                raise Exception(f"Please, return your outputs as a dictionary. Found {output}")

            for key, serializer in serializers.items():
                output[key] = serializer(output[key])

            return output

        run(app, host=host, port=port, log_level="error")
