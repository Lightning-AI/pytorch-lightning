import abc
import base64
from pathlib import Path
from typing import Any, Dict, Optional

import torch
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
            encoded_string = base64.b64encode(image_file.read()).strip()
        return {"image": encoded_string.decode("UTF-8")}


class Number(BaseModel):
    prediction: Optional[int]

    @staticmethod
    def _get_sample_data() -> Dict[Any, Any]:
        return {"prediction": 463}


class ServeBase(LightningWork, abc.ABC):
    def __init__(  # type: ignore
        self,
        host: str = "127.0.0.1",
        port: int = 7777,
        input_type: type = _DefaultInputData,
        output_type: type = _DefaultOutputData,
        **kwargs,
    ):
        super().__init__(parallel=True, host=host, port=port, **kwargs)
        if not issubclass(input_type, BaseModel):
            raise TypeError("input_type must be a pydantic BaseModel class")
        if not issubclass(output_type, BaseModel):
            raise TypeError("output_type must be a pydantic BaseModel class")
        self._supported_pydantic_types = {"string", "number", "integer", "boolean"}
        self._input_type = self._verify_type(input_type)
        self._output_type = self._verify_type(output_type)

    @property
    def device(self):
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def configure_input_type(self) -> type:
        """Override this method to configure the input type for the API.

        By default, it is set to `_DefaultInputData`
        """
        return self._input_type

    def configure_output_type(self) -> type:
        """Override this method to configure the output type for the API.

        By default, it is set to `_DefaultOutputData`
        """
        return self._output_type

    def _verify_type(self, datatype: Any):
        props = datatype.schema()["properties"]
        for k, v in props.items():
            if v["type"] not in self._supported_pydantic_types:
                raise TypeError("Unsupported type")
        return datatype

    def setup(self, *args, **kwargs) -> None:
        """This method is called before the server starts. Override this if you need to download the model or
        initialize the weights, setting up pipelines etc.

        Note that this will be called exactly once on every work machines. So if you have multiple machines for serving,
        this will be called on each of them.
        """
        return

    @abc.abstractmethod
    def infer(self, request: Any) -> Any:
        """This method is called when a request is made to the server.

        This method must be overriden by the user with the prediction logic
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

    def _attach_infer_fn(self, fastapi_app: FastAPI) -> None:
        input_type: type = self.configure_input_type()
        output_type: type = self.configure_output_type()

        def infer_fn(request: input_type):  # type: ignore
            with torch.inference_mode():
                return self.predict(request)

        fastapi_app.post("/predict", response_model=output_type)(infer_fn)

    def _attach_frontend(self, fastapi_app: FastAPI) -> None:
        from lightning_api_access import APIAccessFrontend

        class_name = self.__class__.__name__
        url = self._future_url if self._future_url else self.url
        if not url:
            # if the url is still empty, point it to localhost
            url = f"http://127.0.0.1:{self.port}"
        request, response = {}, {}
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
                    "url": f"{url}/infer",
                    "method": "POST",
                    "request": request,
                    "response": response,
                }
            ]
        )
        fastapi_app.mount("/", StaticFiles(directory=frontend.serve_dir, html=True), name="static")
