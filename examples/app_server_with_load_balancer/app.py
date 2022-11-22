# !pip install torchvision pydantic
import base64
import io

import torch
import torchvision
from PIL import Image
from pydantic import BaseModel

import lightning as L
from lightning.app.components.serve import Image as InputImage
from lightning.app.components.serve import PythonServer


class PyTorchServer(PythonServer):
    def __init__(self):
        super().__init__(
            input_type=InputImage,
            output_type=OutputData,
            cloud_compute=L.CloudCompute("gpu"),
        )

    def setup(self):
        self._model = torchvision.models.resnet18(pretrained=True)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    def predict(self, request):
        image = base64.b64decode(request.image.encode("utf-8"))
        image = Image.open(io.BytesIO(image))
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image = transforms(image)
        image = image.to(self._device)
        prediction = self._model(image.unsqueeze(0))
        return {"prediction": prediction.argmax().item()}


class OutputData(BaseModel):
    prediction: int


app = L.LightningApp(PyTorchServer())


# TODO: name confusion LoadBalancer vs. AutoScaler
# from lightning.app.components import LoadBalancer
# component = LoadBalancer(
#     PyTorchServer,
#     num_replicas=4,
#     balance_function="predict",
#     auto_scale=False,
# )
# app = L.LightningApp(component)
