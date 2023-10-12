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

from lightning.app.core.work import LightningWork
from lightning.app.utilities.imports import _is_gradio_available, requires

if _is_gradio_available():
    import gradio
else:
    gradio = ModuleType("gradio")
    gradio.themes = ModuleType("gradio.themes")

    class __DummyBase:
        pass

    gradio.themes.Base = __DummyBase


class ServeGradio(LightningWork, abc.ABC):
    """The ServeGradio Class enables to quickly create a ``gradio`` based UI for your LightningApp.

    In the example below, the ``ServeGradio`` is subclassed to deploy ``AnimeGANv2``.

    .. literalinclude:: ../../../../examples/app/components/serve/gradio/app.py
        :language: python

    The result would be the following:

    .. image:: https://pl-public-data.s3.amazonaws.com/assets_lightning/anime_gan.gif
        :alt: Animation showing how to AnimeGANv2 UI would looks like.

    """

    inputs: Any
    outputs: Any
    examples: Optional[List] = None
    enable_queue: bool = False
    title: Optional[str] = None
    description: Optional[str] = None

    _start_method = "spawn"

    def __init__(self, *args: Any, theme: Optional[gradio.themes.Base] = None, **kwargs: Any):
        requires("gradio")(super().__init__(*args, **kwargs))
        assert self.inputs
        assert self.outputs
        self._model = None
        self._theme = theme or ServeGradio.__get_lightning_gradio_theme()

        self.ready = False

    @property
    def model(self):
        return self._model

    @abc.abstractmethod
    def predict(self, *args: Any, **kwargs: Any):
        """Override with your logic to make a prediction."""

    @abc.abstractmethod
    def build_model(self) -> Any:
        """Override to instantiate and return your model.

        The model would be accessible under self.model

        """

    def run(self, *args: Any, **kwargs: Any):
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
            theme=self._theme,
        ).launch(
            server_name=self.host,
            server_port=self.port,
            enable_queue=self.enable_queue,
        )

    def configure_layout(self) -> str:
        return self.url

    @staticmethod
    def __get_lightning_gradio_theme():
        return gradio.themes.Default(
            primary_hue=gradio.themes.Color(
                "#ffffff",
                "#e9d5ff",
                "#d8b4fe",
                "#c084fc",
                "#fcfcfc",
                "#a855f7",
                "#9333ea",
                "#8823e1",
                "#6b21a8",
                "#2c2730",
                "#1c1c1c",
            ),
            secondary_hue=gradio.themes.Color(
                "#c3a1e8",
                "#e9d5ff",
                "#d3bbec",
                "#c795f9",
                "#9174af",
                "#a855f7",
                "#9333ea",
                "#6700c2",
                "#000000",
                "#991ef1",
                "#33243d",
            ),
            neutral_hue=gradio.themes.Color(
                "#ede9fe",
                "#ddd6fe",
                "#c4b5fd",
                "#a78bfa",
                "#fafafa",
                "#8b5cf6",
                "#7c3aed",
                "#6d28d9",
                "#6130b0",
                "#8a4ce6",
                "#3b3348",
            ),
        ).set(
            body_background_fill="*primary_50",
            body_background_fill_dark="*primary_950",
            body_text_color_dark="*primary_100",
            body_text_size="*text_sm",
            body_text_color_subdued_dark="*primary_100",
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_950",
            background_fill_secondary="*primary_50",
            background_fill_secondary_dark="*primary_950",
            border_color_accent="*primary_400",
            border_color_accent_dark="*primary_900",
            border_color_primary="*primary_600",
            border_color_primary_dark="*primary_800",
            color_accent="*primary_400",
            color_accent_soft="*primary_300",
            color_accent_soft_dark="*primary_700",
            link_text_color="*primary_500",
            link_text_color_dark="*primary_50",
            link_text_color_active="*secondary_800",
            link_text_color_active_dark="*primary_500",
            link_text_color_hover="*primary_400",
            link_text_color_hover_dark="*primary_400",
            link_text_color_visited="*primary_500",
            link_text_color_visited_dark="*secondary_100",
            block_background_fill="*primary_50",
            block_background_fill_dark="*primary_900",
            block_border_color_dark="*primary_800",
            checkbox_background_color="*primary_50",
            checkbox_background_color_dark="*primary_50",
            checkbox_background_color_focus="*primary_100",
            checkbox_background_color_focus_dark="*primary_100",
            checkbox_background_color_hover="*primary_400",
            checkbox_background_color_hover_dark="*primary_500",
            checkbox_background_color_selected="*primary_300",
            checkbox_background_color_selected_dark="*primary_500",
            checkbox_border_color_dark="*primary_200",
            checkbox_border_radius="*radius_md",
            input_background_fill="*primary_50",
            input_background_fill_dark="*primary_900",
            input_radius="*radius_xxl",
            slider_color="*primary_600",
            slider_color_dark="*primary_700",
            button_large_radius="*radius_xxl",
            button_large_text_size="*text_md",
            button_small_radius="*radius_xxl",
            button_primary_background_fill_dark="*primary_800",
            button_primary_background_fill_hover_dark="*primary_700",
            button_primary_border_color_dark="*primary_800",
            button_secondary_background_fill="*neutral_200",
            button_secondary_background_fill_dark="*primary_600",
        )
