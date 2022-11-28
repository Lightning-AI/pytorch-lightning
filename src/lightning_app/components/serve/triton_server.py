import abc
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import jinja2
import numpy as np
import tritonclient.http as httpclient
import uvicorn
from fastapi import FastAPI
from tritonclient.utils import np_to_triton_dtype

from lightning_app.components.serve.base import ServeBase
from lightning_app.utilities import safe_pickle
from lightning_app.utilities.app_helpers import Logger
from lightning_app.utilities.cloud import is_running_in_cloud
from lightning_app.utilities.packaging.build_config import BuildConfig
from lightning_app.utilities.packaging.cloud_compute import CloudCompute

logger = Logger(__name__)

MODEL_NAME = "lightning-triton"
LIGHTNING_TRITON_BASE_IMAGE = os.getenv("LIGHTNING_TRITON_BASE_IMAGE", "ghcr.io/gridai/lightning-triton:v0.13")


environment = jinja2.Environment()
template = environment.from_string(
    """name: "lightning-triton"
backend: "{{ backend }}"
max_batch_size: {{ max_batch_size }}
default_model_filename: "__lightningapp_triton_model_file.py"

input [
{% for input in inputs %}
  {
    name: "{{ input.name }}"
    data_type: {{ input.type }}
    dims: {{ input.dim }}
  }{{ "," if not loop.last else "" }}
{% endfor %}
]
output [
{% for output in outputs %}
  {
    name: "{{ output.name }}"
    data_type: {{ output.type }}
    dims: {{ output.dim }}
  }{{ "," if not loop.last else "" }}
{% endfor %}
]

instance_group [
  {
    kind: KIND_{{ kind }}
  }
]
"""
)


triton_model_file_template = """
import json
import pickle

import numpy as np
import triton_python_backend_utils as pb_utils


class Request:
    pass


class TritonPythonModel:

    def __init__(self):
        self.work = None
        self.model_config = {}

    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        self.work = pickle.load(open("__model_repository/lightning-triton/1/__lightning_work.pkl", "rb"))
        self.work.setup()

    def execute(self, requests):
        responses = []
        for request in requests:
            req = Request()
            for inp in self.model_config['input']:
                i = pb_utils.get_input_tensor_by_name(request, inp['name'])
                if inp['data_type'] == 'TYPE_STRING':
                    ip = i.as_numpy()[0][0].decode()
                else:
                    ip = i.as_numpy()[0][0]
                setattr(req, inp['name'], ip)
            resp = self.work.infer(req)
            for out in self.model_config['output']:
                if out['name'] not in resp:
                    responses.append(pb_utils.InferenceResponse(
                        output_tensors=[], error=pb_utils.TritonError(f"Output {out['name']} not found in response")))
                    continue
                val = [resp[out['name']]]
                dtype = pb_utils.triton_string_to_numpy(out['data_type'])
                out = pb_utils.Tensor(
                    out['name'],
                    np.array(val, dtype=dtype),
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses
"""


PYDANTIC_TO_NUMPY = {
    "integer": np.int32,
    "number": np.float64,
    "string": np.dtype("object"),
    "boolean": np.dtype("bool"),
}


def pydantic_to_numpy_dtype(pydantic_obj_string):
    return PYDANTIC_TO_NUMPY[pydantic_obj_string]


PYDANTIC_TO_TRITON = {
    "integer": "TYPE_INT32",
    "number": "TYPE_FP64",
    "string": "TYPE_STRING",
    "boolean": "TYPE_BOOL",
}


def pydantic_to_triton_dtype_string(pydantic_obj_string):
    return PYDANTIC_TO_TRITON[pydantic_obj_string]


class TritonServer(ServeBase, abc.ABC):
    def __init__(self, *args, max_batch_size=8, backend="python", **kwargs):
        cloud_build_config = kwargs.get("cloud_build_config", BuildConfig(image=LIGHTNING_TRITON_BASE_IMAGE))
        compute_config = kwargs.get("cloud_compute", CloudCompute("cpu", shm_size=512))
        if compute_config.shm_size < 256:
            raise ValueError("Triton expects the shared memory size (shm_size) to be at least 256MB")
        super().__init__(*args, **kwargs, cloud_build_config=cloud_build_config, cloud_compute=compute_config)
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be greater than 0")
        self.max_batch_size = max_batch_size
        if backend != "python":
            raise ValueError(
                "Currently only python backend is supported. But we are looking for user feedback to"
                "support other backends too. Please reach out to our slack channel or "
                "support@lightning.ai"
            )
        self.backend = backend
        self._triton_server_process = None

    @abc.abstractmethod
    def infer(self, request: Any) -> Any:
        """This method is called when a request is made to the server.

        This method must be overriden by the user with the prediction logic
        """
        pass

    def _attach_triton_proxy_fn(self, fastapi_app: FastAPI):
        input_type: Any = self.configure_input_type()
        output_type: Any = self.configure_output_type()

        client = httpclient.InferenceServerClient(
            url="127.0.0.1:8000", connection_timeout=1200.0, network_timeout=1200.0
        )

        def proxy_fn(request: input_type):  # type: ignore
            # TODO - test with multiple input and multiple output
            inputs = []
            outputs = []
            for property_name, property in input_type.schema()["properties"].items():
                val = [getattr(request, property_name)]
                dtype = pydantic_to_numpy_dtype(property["type"])
                arr = np.array(val, dtype=dtype).reshape((-1, 1))
                data = httpclient.InferInput(property_name, arr.shape, np_to_triton_dtype(arr.dtype))
                data.set_data_from_numpy(arr)
                inputs.append(data)
            for property_name, property in output_type.schema()["properties"].items():
                output = httpclient.InferRequestedOutput(property_name)
                outputs.append(output)
            query_response = client.infer(
                model_name="lightning-triton",
                inputs=inputs,
                outputs=outputs,
                timeout=1200,
            )
            response = {}
            for property_name, property in output_type.schema()["properties"].items():
                # TODO - test with image output if decode is required
                response[property_name] = query_response.as_numpy(property_name).item()
            return response

        fastapi_app.post("/infer", response_model=output_type)(proxy_fn)

    def _get_config_file(self) -> str:
        """Create config.pbtxt file specific for triton-python backend."""
        kind = "GPU" if self.device.type == "cuda" else "CPU"
        input_types = self.configure_input_type()
        output_types = self.configure_output_type()
        inputs = []
        outputs = []
        for k, v in input_types.schema()["properties"].items():
            inputs.append({"name": k, "type": pydantic_to_triton_dtype_string(v["type"]), "dim": "[1]"})
        for k, v in output_types.schema()["properties"].items():
            outputs.append({"name": k, "type": pydantic_to_triton_dtype_string(v["type"]), "dim": "[1]"})
        return template.render(
            kind=kind, inputs=inputs, outputs=outputs, max_batch_size=self.max_batch_size, backend=self.backend
        )

    def _setup_model_repository(self):
        # create the model repository directory
        cwd = Path.cwd()
        if (cwd / "__model_repository").is_dir():
            shutil.rmtree(cwd / "__model_repository")
        repo_path = cwd / f"__model_repository/{MODEL_NAME}/1"
        repo_path.mkdir(parents=True, exist_ok=True)

        # setting the model file
        (repo_path / "__lightningapp_triton_model_file.py").write_text(triton_model_file_template)

        with open(repo_path / "__lightning_work.pkl", "wb+") as f:
            safe_pickle.dump(self, f)

        # setting the config file
        config = self._get_config_file()
        config_path = repo_path.parent
        with open(config_path / "config.pbtxt", "w") as f:
            f.write(config)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run method takes care of configuring and setting up a FastAPI server behind the scenes.

        Normally, you don't need to override this method.
        """
        self._setup_model_repository()

        # setting and exposing the fast api service that sits in front of triton server
        fastapi_app = FastAPI()
        self._attach_triton_proxy_fn(fastapi_app)
        self._attach_frontend(fastapi_app)

        # start triton server in subprocess
        triton_cmd = "tritonserver --model-repository __model_repository"
        if is_running_in_cloud():
            self._triton_server_process = subprocess.Popen(shlex.split(triton_cmd))
        else:
            # locally, installing triton is painful and hence we'll call into docker
            base_image = LIGHTNING_TRITON_BASE_IMAGE
            # TODO - install only if requirements.txt
            cmd = f'bash -c "pip install -r requirements.txt && {triton_cmd}"'
            docker_cmd = shlex.split(
                f"docker run -it --shm-size=256m --rm -p8000:8000 -v {cwd}:/content/ {base_image} {cmd}"
            )
            self._triton_server_process = subprocess.Popen(docker_cmd)

        logger.info(f"Your app has started. View it in your browser: http://{self.host}:{self.port}")
        uvicorn.run(app=fastapi_app, host=self.host, port=self.port, log_level="error")

    def on_exit(self):
        # TODO @sherin add the termination of uvicorn once the issue with signal/uvloop conflict is resolved
        if self._triton_server_process:
            self._triton_server_process.kill()
