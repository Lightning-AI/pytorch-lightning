import abc
from typing import Any, Callable

import uvicorn
from fastapi import Body, FastAPI
from pydantic import BaseModel

from lightning_app.core.work import LightningWork
from lightning_app.utilities.app_helpers import Logger

logger = Logger(__name__)


class ModelServer(LightningWork, abc.ABC):
    def __init__(
        self, host: str = "127.0.0.1", port: int = 7777, input_type: type = str, output_type: type = str, **kwargs
    ):
        """The ModelServer Class enables to easily get your machine learning server up and running.

        Arguments:
            host: Address to be used for running the server.
            port: Port to be used to running the server.
            input_type: Optional `input_type` to be provided.
                This can either be basic python datatypes, or a pydantic BaseModel class.
            output_type: Optional `output_type` to be provided.
                This can either be basic python datatypes, or a pydantic BaseModel class.

        .. doctest::

            >>> from lightning.app.components.serve.model_server import ModelServer
            >>> from pydantic import BaseModel
            >>>
            >>> class InputData(BaseModel):
            ...     image: str
            ...
            >>> class OutputData(BaseModel):
            ...     prediction: str
            ...
            >>> class SimpleServer(ModelServer):
            ...     def setup(self):
            ...         self._model = lambda x: x + " " + x
            ...     def predict(self, request):
            ...         return {"prediction": self._model(request.image)}
            ...
            >>> app = SimpleServer(input_type=InputData, output_type=OutputData)
        """
        super().__init__(parallel=True, host=host, port=port, **kwargs)
        self._input_type = input_type
        self._output_type = output_type

    def setup(self):
        """This method is called before the server starts. Override this if you need to download the model or
        initialize the weights, setting up pipelines etc.

        Note that this will be called exactly once on every work machines. So if you have multiple machines for serving,
        this will be called on each of them.
        """
        return

    def configure_input_type(self) -> type:
        return self._input_type

    def configure_output_type(self) -> type:
        return self._output_type

    @abc.abstractmethod
    def predict(self, request: Any):
        """This method is called when a request is made to the server.

        This method must be overriden by the user with the prediction logic. The pre/post processing, actual prediction
        using the model(s) etc goes here
        """
        pass

    def _attach_predict_fn(self, fastapi_app: FastAPI):
        input_type: type = self.configure_input_type()
        output_type: type = self.configure_output_type()

        if not issubclass(input_type, BaseModel):
            # adding a Body() default value is necessary for FastAPI to add the request to body
            # instead of query params
            def predict_fn(request: input_type = Body()):
                return self.predict(request)

        else:

            def predict_fn(request: input_type):
                return self.predict(request)

        fastapi_app.post("/predict", response_model=output_type)(predict_fn)

    def run(self):
        """Run method takes care of configuring and setting up a FastAPI server behind the scenes.

        Normally, you don't need to override this method.
        """
        self.setup()

        fastapi_app = FastAPI()
        self._attach_predict_fn(fastapi_app)

        logger.info(f"Your app has started. View it in your browser: http://{self.host}:{self.port}")
        uvicorn.run(app=fastapi_app, host=self.host, port=self.port, log_level="error")
