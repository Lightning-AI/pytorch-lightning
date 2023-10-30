# !pip install torchvision
from lightning.app import LightningApp, CloudCompute
from lightning.app.components.serve import PythonServer, Image, Number
import base64, io, torchvision, torch
from PIL import Image as PILImage


class PyTorchServer(PythonServer):
    def setup(self):
        self._model = torchvision.models.resnet18(pretrained=True)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    def predict(self, request):
        image = base64.b64decode(request.image.encode("utf-8"))
        image = PILImage.open(io.BytesIO(image))
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = transforms(image)
        image = image.to(self._device)
        prediction = self._model(image.unsqueeze(0))
        return {"prediction": prediction.argmax().item()}


component = PyTorchServer(
   input_type=Image, output_type=Number, cloud_compute=CloudCompute('gpu')
)
app = LightningApp(component)
