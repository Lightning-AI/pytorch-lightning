import time
from multiprocessing import Process
from typing import Any, Dict, Optional

import requests
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.serve.module import ServableModule
from pytorch_lightning.utilities.imports import _RequirementAvailable
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class SanityServing(Callback):

    """This callback enables to validate a model is servable following the ServableModule API."""

    def __init__(
        self,
        optimization: Optional[str] = None,
        server: str = "fastapi",
        host: str = "127.0.0.1",
        port: int = 8080,
    ):
        super().__init__()
        if not _RequirementAvailable("fastapi"):
            raise ModuleNotFoundError("The package fastapi is required by the SanityServing callback.")
        if not _RequirementAvailable("uvicorn"):
            raise ModuleNotFoundError("The package uvicorn is required by the SanityServing callback.")

        # TODO: Add support for those optimizations
        assert optimization in (None, "trace", "script", "onnx", "tensor_rt")
        if optimization in ("trace", "script", "onnx", "tensor_rt"):
            raise NotImplementedError(f"The optimization {optimization} is currently not supported.")

        # TODO: Add support for testing with those server services
        assert server in ("fastapi", "ml_server", "torch_serve", "sagemaker")
        if server != "fastapi":
            raise NotImplementedError("Only the fastapi server is currently supported.")

        self.optimization = optimization
        self.host = host
        self.port = port
        self.server = server
        self.resp = None

    @rank_zero_only
    def on_train_start(self, trainer: "pl.Trainer", servable_model: "ServableModule"):
        isinstance(servable_model, ServableModule)

        process = Process(target=self._start_server, args=(servable_model, self.host, self.port, self.optimization))
        process.start()

        ready = False
        while not ready:
            try:
                resp = requests.get(f"http://{self.host}:{self.port}/ping")
                ready = resp.status_code == 200
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(0.1)

        payload = servable_model.configure_payload()
        self.resp = requests.post(f"http://{self.host}:{self.port}/serve", json=payload)
        process.kill()

    @property
    def successful(self) -> bool:
        return self.resp.status_code == 200 if self.resp else False

    def state_dict(self):
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
