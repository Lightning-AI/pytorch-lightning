from typing import Any, List

import torch
import torchvision
from pydantic import BaseModel

import lightning as L


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
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = torchvision.models.resnet18(pretrained=True).to(self._device)

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
            image = L.app.components.serve.types.image.Image.deserialize(request.image)
            image = transforms(image).unsqueeze(0)
            images.append(image)
        images = torch.cat(images)
        images = images.to(self._device)
        predictions = self._model(images)
        results = predictions.argmax(1).cpu().numpy().tolist()
        return BatchResponse(outputs=[{"prediction": pred} for pred in results])


class MyAutoScaler(L.app.components.AutoScaler):
    def scale(self, replicas: int, metrics: dict) -> int:
        """The default scaling logic that users can override.

        Args:
            replicas: The number of running works.
            metrics: ``metrics['pending_requests']`` is the total number of requests that are currently pending.
                ``metrics['pending_works']`` is the number of pending works.

        Returns:
            The target number of running works. The value will be adjusted after this method runs
            so that it satisfies ``min_replicas<=replicas<=max_replicas``.
        """
        max_requests_per_work = self.max_batch_size
        min_requests_per_work = 1
        pending_requests_per_running_or_pending_work = metrics["pending_requests"] / (
            replicas + metrics["pending_works"]
        )

        # upscale
        if pending_requests_per_running_or_pending_work >= max_requests_per_work:
            return replicas + 1

        # downscale
        if pending_requests_per_running_or_pending_work < min_requests_per_work:
            return replicas - 1

        return replicas


app = L.LightningApp(
    MyAutoScaler(
        PyTorchServer,
        min_replicas=2,
        max_replicas=4,
        autoscale_interval=10,
        endpoint="predict",
        input_type=RequestModel,
        output_type=Any,
        timeout_batching=1,
    )
)
