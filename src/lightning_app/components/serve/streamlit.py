import abc
from typing import Any

from lightning_app import LightningWork

# if _is_streamlit_available():
#     import gradio
# else:
#     streamlit = ModuleType("gradio")


class ServeStreamlit(LightningWork, abc.ABC):

    # inputs: Any
    # outputs: Any
    # examples: Optional[List] = None
    # enable_queue: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # requires("gradio")()
        # assert self.inputs
        # assert self.outputs
        # self._model = None

    @property
    def model(self):
        return self._model

    @abc.abstractmethod
    def render(self):
        """Override with your streamlit render."""

    @abc.abstractmethod
    def build_model(self) -> Any:
        """Override to instantiate and return your model.

        The model will be accessible under ``self.model``.
        """

    def run(self, *args, **kwargs):
        pass
