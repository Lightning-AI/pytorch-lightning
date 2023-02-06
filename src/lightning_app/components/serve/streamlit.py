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
import inspect
import os
import pydoc
import subprocess
import sys
from typing import Any, Callable, Type

from lightning_app.core.work import LightningWork
from lightning_app.utilities.app_helpers import StreamLitStatePlugin
from lightning_app.utilities.state import AppState


class ServeStreamlit(LightningWork, abc.ABC):
    """The ``ServeStreamlit`` work allows you to use streamlit from a work.

    You can optionally build a model in the ``build_model`` hook, which will only be called once per session.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ready = False

        self._process = None

    @property
    def model(self) -> Any:
        return getattr(self, "_model", None)

    @abc.abstractmethod
    def render(self) -> None:
        """Override with your streamlit render function."""

    def build_model(self) -> Any:
        """Optionally override to instantiate and return your model.

        The model will be accessible under ``self.model``.
        """
        return None

    def run(self) -> None:
        env = os.environ.copy()
        env["LIGHTNING_COMPONENT_NAME"] = self.name
        env["LIGHTNING_WORK"] = self.__class__.__name__
        env["LIGHTNING_WORK_MODULE_FILE"] = inspect.getmodule(self).__file__
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                __file__,
                "--server.address",
                str(self.host),
                "--server.port",
                str(self.port),
                "--server.headless",
                "true",  # do not open the browser window when running locally
            ],
            env=env,
        )
        self.ready = True
        self._process.wait()

    def on_exit(self) -> None:
        if self._process is not None:
            self._process.kill()

    def configure_layout(self) -> str:
        return self.url


class _PatchedWork:
    """The ``_PatchedWork`` is used to emulate a work instance from a subprocess. This is acheived by patching the
    self reference in methods an properties to point to the AppState.

    Args:
        state: The work state to patch
        work_class: The work class to emulate
    """

    def __init__(self, state: AppState, work_class: Type):
        super().__init__()
        self._state = state
        self._work_class = work_class

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self._state, name)
        except AttributeError:
            # The name isn't in the state, so check if it's a callable or a property
            attribute = inspect.getattr_static(self._work_class, name)
            if callable(attribute):
                attribute = attribute.__get__(self, self._work_class)
                return attribute
            elif isinstance(attribute, (staticmethod, property)):
                return attribute.__get__(self, self._work_class)

            # Look for the name in the instance (e.g. for private variables)
            return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ["_state", "_work_class"]:
            return object.__setattr__(self, name, value)

        if hasattr(self._state, name):
            return setattr(self._state, name, value)
        return object.__setattr__(self, name, value)


def _reduce_to_component_scope(state: AppState, component_name: str) -> AppState:
    """Given the app state, this utility traverses down to the level of the given component name."""
    component_name_parts = component_name.split(".")[1:]  # exclude root
    component_state = state
    for part in component_name_parts:
        component_state = getattr(component_state, part)
    return component_state


def _get_work_class() -> Callable:
    """Import the work class specified in the environment."""
    work_name = os.environ["LIGHTNING_WORK"]
    work_module_file = os.environ["LIGHTNING_WORK_MODULE_FILE"]
    module = pydoc.importfile(work_module_file)
    return getattr(module, work_name)


def _build_model(work: ServeStreamlit) -> None:
    import streamlit as st

    # Build the model (once per session, equivalent to gradio when enable_queue is Flase)
    if "_model" not in st.session_state:
        with st.spinner("Building model..."):
            st.session_state["_model"] = work.build_model()

    work._model = st.session_state["_model"]


def _main() -> None:
    # Get the AppState
    app_state = AppState(plugin=StreamLitStatePlugin())
    work_state = _reduce_to_component_scope(app_state, os.environ["LIGHTNING_COMPONENT_NAME"])

    # Create the patched work
    work_class = _get_work_class()
    patched_work = _PatchedWork(work_state, work_class)

    # Build and attach the model
    _build_model(patched_work)

    # Render
    patched_work.render()


if __name__ == "__main__":
    _main()
