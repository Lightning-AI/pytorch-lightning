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

import abc
import inspect
import os
import pydoc
import subprocess
import sys
from typing import Any, Callable, Optional

import fastapi  # noqa E511
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from lightning.app.components.serve.types import _DESERIALIZER, _SERIALIZER
from lightning.app.core.work import LightningWork
from lightning.app.utilities.app_helpers import Logger

logger = Logger(__name__)


fastapi_service = FastAPI()


class _InferenceCallable:
    def __init__(
        self,
        deserialize: Callable,
        predict: Callable,
        serialize: Callable,
    ):
        self.deserialize = deserialize
        self.predict = predict
        self.serialize = serialize

    async def run(self, data) -> Any:
        return self.serialize(self.predict(self.deserialize(data)))


class ModelInferenceAPI(LightningWork, abc.ABC):
    def __init__(
        self,
        input: Optional[str] = None,
        output: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 7777,
        workers: int = 0,
    ):
        """The ModelInferenceAPI Class enables to easily get your model served.

        Arguments:
            input: Optional `input` to be provided. This would make provide a built-in deserializer.
            output: Optional `output` to be provided. This would make provide a built-in serializer.
            host: Address to be used to serve the model.
            port: Port to be used to serve the model.
            workers: Number of workers for the uvicorn. Warning, this won't work if your subclass takes more arguments.
        """
        super().__init__(parallel=True, host=host, port=port)
        if input and input not in _DESERIALIZER:
            raise Exception(f"Only input in {_DESERIALIZER.keys()} are supported.")
        if output and output not in _SERIALIZER:
            raise Exception(f"Only output in {_SERIALIZER.keys()} are supported.")
        self.input = input
        self.output = output
        self.workers = workers
        self._model = None

        self.ready = False

    @property
    def model(self):
        return self._model

    @abc.abstractmethod
    def build_model(self) -> Any:
        """Override to define your model."""

    def deserialize(self, data) -> Any:
        return data

    @abc.abstractmethod
    def predict(self, data) -> Any:
        """Override to add your predict logic."""

    def serialize(self, data) -> Any:
        return data

    def run(self):
        global fastapi_service
        if self.workers > 1:
            # TODO: This is quite limitated
            # Find a more reliable solution to enable multi workers serving.
            env = os.environ.copy()
            module = inspect.getmodule(self).__file__
            env["LIGHTNING_MODEL_INFERENCE_API_FILE"] = module
            env["LIGHTNING_MODEL_INFERENCE_API_CLASS_NAME"] = self.__class__.__name__
            if self.input:
                env["LIGHTNING_MODEL_INFERENCE_API_INPUT"] = self.input
            if self.output:
                env["LIGHTNING_MODEL_INFERENCE_API_OUTPUT"] = self.output
            command = [
                sys.executable,
                "-m",
                "uvicorn",
                "--workers",
                str(self.workers),
                "--host",
                str(self.host),
                "--port",
                str(self.port),
                "serve:fastapi_service",
            ]
            process = subprocess.Popen(command, env=env, cwd=os.path.dirname(__file__))
            self.ready = True
            process.wait()
        else:
            self._populate_app(fastapi_service)
            self.ready = True
            self._launch_server(fastapi_service)

    def _populate_app(self, fastapi_service: FastAPI):
        self._model = self.build_model()

        fastapi_service.post("/predict", response_class=JSONResponse)(
            _InferenceCallable(
                deserialize=_DESERIALIZER[self.input] if self.input else self.deserialize,
                predict=self.predict,
                serialize=_SERIALIZER[self.output] if self.output else self.serialize,
            ).run
        )

    def _launch_server(self, fastapi_service: FastAPI):
        logger.info(f"Your app has started. View it in your browser: http://{self.host}:{self.port}")
        uvicorn.run(app=fastapi_service, host=self.host, port=self.port, log_level="error")

    def configure_layout(self) -> str:
        return f"{self.url}/docs"


def _maybe_create_instance() -> Optional[ModelInferenceAPI]:
    """This function tries to re-create the user `ModelInferenceAPI` if the environment associated with multi
    workers are present."""
    render_fn_name = os.getenv("LIGHTNING_MODEL_INFERENCE_API_CLASS_NAME", None)
    render_fn_module_file = os.getenv("LIGHTNING_MODEL_INFERENCE_API_FILE", None)
    if render_fn_name is None or render_fn_module_file is None:
        return None
    module = pydoc.importfile(render_fn_module_file)
    cls = getattr(module, render_fn_name)
    input = os.getenv("LIGHTNING_MODEL_INFERENCE_API_INPUT", None)
    output = os.getenv("LIGHTNING_MODEL_INFERENCE_API_OUTPUT", None)
    return cls(input=input, output=output)


instance = _maybe_create_instance()
if instance:
    instance._populate_app(fastapi_service)
