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

import abc
from functools import partial
from types import ModuleType
from typing import Any, List, Optional

from lightning_app.core.work import LightningWork
from lightning_app.utilities.imports import _is_gradio_available, requires

if _is_gradio_available():
    import gradio
else:
    gradio = ModuleType("gradio")


class ServeGradio(LightningWork, abc.ABC):
    """The ServeGradio Class enables to quickly create a ``gradio`` based UI for your LightningApp.

    In the example below, the ``ServeGradio`` is subclassed to deploy ``AnimeGANv2``.

    .. literalinclude:: ../../../examples/app_components/serve/gradio/app.py
        :language: python

    The result would be the following:

    .. image:: https://pl-flash-data.s3.amazonaws.com/assets_lightning/anime_gan.gif
        :alt: Animation showing how to AnimeGANv2 UI would looks like.
    """

    inputs: Any
    outputs: Any
    examples: Optional[List] = None
    enable_queue: bool = False
    title: Optional[str] = None
    description: Optional[str] = None

    _start_method = "spawn"

    def __init__(self, *args, **kwargs):
        requires("gradio")(super().__init__(*args, **kwargs))
        assert self.inputs
        assert self.outputs
        self._model = None

        self.ready = False

    @property
    def model(self):
        return self._model

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """Override with your logic to make a prediction."""

    @abc.abstractmethod
    def build_model(self) -> Any:
        """Override to instantiate and return your model.

        The model would be accessible under self.model
        """

    def run(self, *args, **kwargs):
        if self._model is None:
            self._model = self.build_model()
        fn = partial(self.predict, *args, **kwargs)
        fn.__name__ = self.predict.__name__
        self.ready = True
        gradio.Interface(
            fn=fn,
            inputs=self.inputs,
            outputs=self.outputs,
            examples=self.examples,
            title=self.title,
            description=self.description,
        ).launch(
            server_name=self.host,
            server_port=self.port,
            enable_queue=self.enable_queue,
        )

    def configure_layout(self) -> str:
        return self.url
