import base64
from io import BytesIO

from lightning_app.components.serve.types.type import BaseType
from lightning_app.utilities.imports import _is_pil_available, _is_torch_available

if _is_torch_available():
    from torch import Tensor

if _is_pil_available():
    from PIL import Image as PILImage


class Image(BaseType):
    @staticmethod
    def deserialize(data: dict):
        encoded_with_padding = (data + "===").encode("ascii")
        img = base64.b64decode(encoded_with_padding)
        buffer = BytesIO(img)
        return PILImage.open(buffer, mode="r")

    @staticmethod
    def serialize(tensor: "Tensor") -> str:
        tensor = tensor.squeeze(0).numpy()
        print(tensor.shape)
        image = PILImage.fromarray(tensor)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        encoded = buffer.getvalue()
        return base64.b64encode(encoded).decode("ascii")
