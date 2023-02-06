# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
