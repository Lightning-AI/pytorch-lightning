import time
from modulefinder import Module
from multiprocessing import Process
from typing import Any, Callable, Dict, Tuple

import requests
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.imports import requires
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class ServableModule(Module):
    def configure_payload(self) -> Dict[str, Any]:
        ...

    def configure_inputs_outputs(self) -> Tuple[Dict[str, Callable], Dict[str, Callable]]:
        ...

    def serve_step(self, *args, **kwargs):
        ...


def _start_server(servable_model: ServableModule, host: str, port: int) -> None:
    """This method starts a simple FastAPI server with a predict and ping endpoint."""
    from fastapi import Body, FastAPI
    from uvicorn import run

    app = FastAPI()

    inputs, outputs = servable_model.configure_inputs_outputs()

    @app.get("/ping")
    def ping() -> bool:
        return True

    @app.post("/predict")
    async def predict(payload: dict = Body(...)) -> Dict[str, Any]:
        body = payload["body"]

        for key, deserializer in inputs.items():
            body[key] = deserializer(body[key])

        with torch.no_grad():
            output = servable_model.serve_step(**body)
        assert isinstance(output, dict)

        for key, serializer in outputs.items():
            output[key] = serializer(output[key])

        return output

    run(app, host=host, port=port, log_level="error")


class SanityServing(Callback):

    """This callback enables to validate a model is servable following the ServableModule API."""

    @requires("fastapi")
    @requires("uvicorn")
    def __init__(self):
        super().__init__()

    @rank_zero_only
    def on_train_start(self, trainer: "pl.Trainer", servable_model: "ServableModule"):
        isinstance(servable_model, ServableModule)

        process = Process(target=_start_server, args=(servable_model, "127.0.0.1", 8080))
        process.start()

        ready = False
        while not ready:
            try:
                resp = requests.get("http://127.0.0.1:8080/ping")
                ready = resp.status_code == 200
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(0.1)

        payload = servable_model.configure_payload()
        resp = requests.post("http://127.0.0.1:8080/predict", json=payload)
        process.kill()
        assert resp.json() == {"output": [-7.451034069061279, 1.635885238647461]}
