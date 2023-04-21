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

from functools import partial

import gradio as gr
import requests
import torch
from PIL import Image

import lightning as L
from lightning.app.components import ServeGradio


# Credit to @akhaliq for his inspiring work.
# Find his original code there: https://huggingface.co/spaces/akhaliq/AnimeGANv2/blob/main/app.py
class AnimeGANv2UI(ServeGradio):
    inputs = gr.inputs.Image(type="pil")
    outputs = gr.outputs.Image(type="pil")
    elon = "https://upload.wikimedia.org/wikipedia/commons/3/34/Elon_Musk_Royal_Society_%28crop2%29.jpg"
    img = Image.open(requests.get(elon, stream=True).raw)
    img.save("elon.jpg")
    examples = [["elon.jpg"]]

    def __init__(self):
        super().__init__()
        self.ready = False

    def predict(self, img):
        return self.model(img=img)

    def build_model(self):
        repo = "AK391/animegan2-pytorch:main"
        model = torch.hub.load(repo, "generator", device="cpu")
        face2paint = torch.hub.load(repo, "face2paint", size=512, device="cpu")
        self.ready = True
        return partial(face2paint, model=model)


class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.demo = AnimeGANv2UI()

    def run(self):
        self.demo.run()

    def configure_layout(self):
        tabs = []
        if self.demo.ready:
            tabs.append({"name": "Home", "content": self.demo})
        return tabs


app = L.LightningApp(RootFlow())
