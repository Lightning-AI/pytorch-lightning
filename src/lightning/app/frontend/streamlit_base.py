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

"""This file gets run by streamlit, which we launch within Lightning.

From here, we will call the render function that the user provided in ``configure_layout``.
"""
import os
import pydoc
from typing import Callable

from lightning.app.frontend.utils import _reduce_to_flow_scope
from lightning.app.utilities.app_helpers import StreamLitStatePlugin
from lightning.app.utilities.state import AppState


def _get_render_fn_from_environment() -> Callable:
    render_fn_name = os.environ["LIGHTNING_RENDER_FUNCTION"]
    render_fn_module_file = os.environ["LIGHTNING_RENDER_MODULE_FILE"]
    module = pydoc.importfile(render_fn_module_file)
    return getattr(module, render_fn_name)


def _main():
    """Run the render_fn with the current flow_state."""
    app_state = AppState(plugin=StreamLitStatePlugin())

    # Fetch the information of which flow attaches to this streamlit instance
    flow_state = _reduce_to_flow_scope(app_state, flow=os.environ["LIGHTNING_FLOW_NAME"])

    # Call the provided render function.
    # Pass it the state, scoped to the current flow.
    render_fn = _get_render_fn_from_environment()
    render_fn(flow_state)


if __name__ == "__main__":
    _main()
