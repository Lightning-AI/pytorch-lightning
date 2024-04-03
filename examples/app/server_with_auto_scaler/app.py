# ! pip install torch torchvision
from typing import List

import torch
import torchvision
from lightning.app import CloudCompute, LightningApp
from pydantic import BaseModel


class BatchRequestModel(BaseModel):
    inputs: List[app.components.Image]


class BatchResponse(BaseModel):
    outputs: List[app.components.Number]


class PyTorchServer(app.components.PythonServer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            input_type=BatchRequestModel,
            output_type=BatchResponse,
            *args,
            **kwargs,
        )

    def setup(self):
        if torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        self._model = torchvision.models.resnet18(pretrained=True).to(self._device)

    def predict(self, requests: BatchRequestModel):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        images = []
        for request in requests.inputs:
            image = app.components.serve.types.image.Image.deserialize(request.image)
            image = transforms(image).unsqueeze(0)
            images.append(image)
        images = torch.cat(images)
        images = images.to(self._device)
        predictions = self._model(images)
        results = predictions.argmax(1).cpu().numpy().tolist()
        return BatchResponse(outputs=[{"prediction": pred} for pred in results])


class MyAutoScaler(app.components.AutoScaler):
    def scale(self, replicas: int, metrics: dict) -> int:
        pending_requests = metrics["pending_requests"]
        active_or_pending_works = replicas + metrics["pending_works"]

        if active_or_pending_works == 0:
            return 1 if pending_requests > 0 else 0

        pending_requests_per_running_or_pending_work = pending_requests / active_or_pending_works

        # scale out if the number of pending requests exceeds max batch size.
        max_requests_per_work = self.max_batch_size
        if pending_requests_per_running_or_pending_work >= max_requests_per_work:
            return replicas + 1

        # scale in if the number of pending requests is below 25% of max_requests_per_work
        min_requests_per_work = max_requests_per_work * 0.25
        if pending_requests_per_running_or_pending_work < min_requests_per_work:
            return replicas - 1

        return replicas


app = LightningApp(
    MyAutoScaler(
        # work class and args
        PyTorchServer,
        cloud_compute=CloudCompute("gpu"),
        # autoscaler specific args
        min_replicas=1,
        max_replicas=4,
        scale_out_interval=10,
        scale_in_interval=10,
        endpoint="predict",
        input_type=app.components.Image,
        output_type=app.components.Number,
        timeout_batching=1,
        max_batch_size=8,
    )
)
