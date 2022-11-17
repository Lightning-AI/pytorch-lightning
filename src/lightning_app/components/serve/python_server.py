import abc
import base64
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

from lightning_app.core.work import LightningWork
from lightning_app.utilities.app_helpers import Logger

logger = Logger(__name__)


class _DefaultInputData(BaseModel):
    payload: str


class _DefaultOutputData(BaseModel):
    prediction: str


class Image(BaseModel):
    image: Optional[str]

    @staticmethod
    def _get_sample_data() -> Dict[Any, Any]:
        imagepath = Path(__file__).parent / "catimage.png"
        with open(imagepath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return {"image": encoded_string.decode("UTF-8")}


class Number(BaseModel):
    prediction: Optional[int]

    @staticmethod
    def _get_sample_data() -> Dict[Any, Any]:
        return {"prediction": 463}


class PythonServer(LightningWork, abc.ABC):
    def __init__(  # type: ignore
        self,
        host: str = "127.0.0.1",
        port: int = 7777,
        input_type: type = _DefaultInputData,
        output_type: type = _DefaultOutputData,
        **kwargs,
    ):
        """The PythonServer Class enables to easily get your machine learning server up and running.

        Arguments:
            host: Address to be used for running the server.
            port: Port to be used to running the server.
            input_type: Optional `input_type` to be provided. This needs to be a pydantic BaseModel class.
                The default data type is good enough for the basic usecases and it expects the data
                to be a json object that has one key called `payload`

                ```
                input_data = {"payload": "some data"}
                ```

                and this can be accessed as `request.payload` in the `predict` method.

                ```
                def predict(self, request):
                    data = request.payload
                ```

            output_type: Optional `output_type` to be provided. This needs to be a pydantic BaseModel class.
                The default data type is good enough for the basic usecases. It expects the return value of
                the `predict` method to be a dictionary with one key called `prediction`.

                ```
                def predict(self, request):
                    # some code
                    return {"prediction": "some data"}
                ```

                and this can be accessed as `response.json()["prediction"]` in the client if
                you are using requests library

        .. doctest::

            >>> from lightning_app.components.serve.python_server import PythonServer
            >>> from lightning_app import LightningApp
            >>>
            ...
            >>> class SimpleServer(PythonServer):
            ...     def setup(self):
            ...         self._model = lambda x: x + " " + x
            ...     def predict(self, request):
            ...         return {"prediction": self._model(request.image)}
            ...
            >>> app = LightningApp(SimpleServer())
        """
        super().__init__(parallel=True, host=host, port=port, **kwargs)
        if not issubclass(input_type, BaseModel):
            raise TypeError("input_type must be a pydantic BaseModel class")
        if not issubclass(output_type, BaseModel):
            raise TypeError("output_type must be a pydantic BaseModel class")
        self._input_type = input_type
        self._output_type = output_type

    def setup(self) -> None:
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
    def predict(self, request: Any) -> Any:
        """This method is called when a request is made to the server.

        This method must be overriden by the user with the prediction logic. The pre/post processing, actual prediction
        using the model(s) etc goes here
        """
        pass

    @staticmethod
    def _get_sample_dict_from_datatype(datatype: Any) -> dict:
        if hasattr(datatype, "_get_sample_data"):
            return datatype._get_sample_data()

        datatype_props = datatype.schema()["properties"]
        out: Dict[str, Any] = {}
        for k, v in datatype_props.items():
            if v["type"] == "string":
                out[k] = "data string"
            elif v["type"] == "number":
                out[k] = 0.0
            elif v["type"] == "integer":
                out[k] = 0
            elif v["type"] == "boolean":
                out[k] = False
            else:
                raise TypeError("Unsupported type")
        return out

    def _attach_predict_fn(self, fastapi_app: FastAPI) -> None:
        input_type: type = self.configure_input_type()
        output_type: type = self.configure_output_type()

        def predict_fn(request: input_type):  # type: ignore
            return self.predict(request)

        fastapi_app.post("/predict", response_model=output_type)(predict_fn)

    def _attach_frontend(self, fastapi_app: FastAPI) -> None:
        from lightning_api_access import APIAccessFrontend

        class_name = self.__class__.__name__
        url = self._future_url if self._future_url else self.url
        if not url:
            # if the url is still empty, point it to localhost
            url = f"http://127.0.0.1:{self.port}"
        url = f"{url}/predict"
        datatype_parse_error = False
        try:
            request = self._get_sample_dict_from_datatype(self.configure_input_type())
        except TypeError:
            datatype_parse_error = True

        try:
            response = self._get_sample_dict_from_datatype(self.configure_output_type())
        except TypeError:
            datatype_parse_error = True

        if datatype_parse_error:

            @fastapi_app.get("/")
            def index() -> str:
                return (
                    "Automatic generation of the UI is only supported for simple, "
                    "non-nested datatype with types string, integer, float and boolean"
                )

            return

        frontend = APIAccessFrontend(
            apis=[
                {
                    "name": class_name,
                    "url": url,
                    "method": "POST",
                    "request": request,
                    "response": response,
                }
            ]
        )
        fastapi_app.mount("/", StaticFiles(directory=frontend.serve_dir, html=True), name="static")

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run method takes care of configuring and setting up a FastAPI server behind the scenes.

        Normally, you don't need to override this method.
        """
        self.setup()

        fastapi_app = FastAPI()
        self._attach_predict_fn(fastapi_app)
        self._attach_frontend(fastapi_app)

        logger.info(f"Your app has started. View it in your browser: http://{self.host}:{self.port}")
        uvicorn.run(app=fastapi_app, host=self.host, port=self.port, log_level="error")
