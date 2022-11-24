import base64
import io
from typing import Any, List

import torch
import torchvision
from PIL import Image as PILImage
from pydantic import BaseModel

import lightning as L
from lightning.app.components import AutoScaler
from lightning.app.components.serve import PythonServer
from lightning.app.utilities.network import find_free_network_port


class RequestModel(BaseModel):
    image: str


class BatchRequestModel(BaseModel):
    inputs: List[RequestModel]


class BatchResponse(BaseModel):
    outputs: List[Any]


class PyTorchServer(PythonServer):
    def setup(self):
        self._model = torchvision.models.resnet18(pretrained=True)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    def predict(self, requests: BatchRequestModel):
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        images = []
        for request in requests.inputs:
            image = base64.b64decode(request.image.encode("utf-8"))
            image = PILImage.open(io.BytesIO(image))
            image = transforms(image).unsqueeze(0)
            images.append(image)
        images = torch.cat(images)
        images = images.to(self._device)
        predictions = self._model(images)
        results = predictions.argmax(1).cpu().numpy().tolist()
        return BatchResponse(outputs=[{"prediction": e} for e in results])


class RootFlow(AutoScaler):
    def create_worker(self, *args, **kwargs) -> L.LightningWork:
        return PyTorchServer(
            port=find_free_network_port(),
            input_type=BatchRequestModel,
            output_type=BatchResponse,
            cloud_compute=L.CloudCompute("gpu"),
        )


app = L.LightningApp(
    RootFlow(
        input_schema=RequestModel,
        output_schema=Any,
        batch_timeout_secs=0.1,
        worker_url="predict",
    )
)
