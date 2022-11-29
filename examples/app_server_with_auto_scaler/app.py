import warnings  # FIXME: remove before merge
from typing import Any, List

import torch
import torchvision
from pydantic import BaseModel

import lightning as L

warnings.filterwarnings("ignore", "Arguments other than a weight enum", UserWarning)  # FIXME: remove before merge
warnings.filterwarnings("ignore", "The parameter 'pretrained' is deprecated", UserWarning)  # FIXME: remove before merge


class RequestModel(BaseModel):
    image: str


class BatchRequestModel(BaseModel):
    inputs: List[RequestModel]


class BatchResponse(BaseModel):
    outputs: List[Any]


class PyTorchServer(L.app.components.PythonServer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            port=L.app.utilities.network.find_free_network_port(),
            input_type=BatchRequestModel,
            output_type=BatchResponse,
            cloud_compute=L.CloudCompute("gpu"),
        )

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
            image = L.app.components.Image.deserialize(request.image)
            image = transforms(image).unsqueeze(0)
            images.append(image)
        images = torch.cat(images)
        images = images.to(self._device)
        predictions = self._model(images)
        results = predictions.argmax(1).cpu().numpy().tolist()
        return BatchResponse(outputs=[{"prediction": e} for e in results])


class MyAutoScaler(L.app.components.AutoScaler):
    def scale(self, replicas, metrics):
        """The default replication logic that users can override."""

        # upscale
        if metrics["pending_requests"] > self.upscale_threshold * metrics["pending_works"]:
            return replicas + 1

        # downscale
        if metrics["pending_requests"] < self.downscale_threshold:
            return replicas - 1

        return replicas


app = L.LightningApp(
    MyAutoScaler(
        PyTorchServer,
        min_replicas=2,
        max_replicas=4,
        worker_url="predict",
        input_schema=RequestModel,
        output_schema=Any,
        timeout_batch=1,
        autoscale_interval=10,
    )
)
