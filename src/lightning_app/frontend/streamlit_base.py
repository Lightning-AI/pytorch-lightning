"""This file gets run by streamlit, which we launch within Lightning.

From here, we will call the render function that the user provided in ``configure_layout``.
"""
import os
import pydoc
from typing import Callable, Union

from lightning_app.core.flow import LightningFlow
from lightning_app.utilities.app_helpers import StreamLitStatePlugin
from lightning_app.utilities.state import AppState

app_state = AppState(plugin=StreamLitStatePlugin())


def _get_render_fn_from_environment() -> Callable:
    render_fn_name = os.environ["LIGHTNING_RENDER_FUNCTION"]
    render_fn_module_file = os.environ["LIGHTNING_RENDER_MODULE_FILE"]
    module = pydoc.importfile(render_fn_module_file)
    return getattr(module, render_fn_name)


def _app_state_to_flow_scope(state: AppState, flow: Union[str, LightningFlow]) -> AppState:
    """Returns a new AppState with the scope reduced to the given flow, as if the given flow as the root."""
    flow_name = flow.name if isinstance(flow, LightningFlow) else flow
    flow_name_parts = flow_name.split(".")[1:]  # exclude root
    flow_state = state
    for part in flow_name_parts:
        flow_state = getattr(flow_state, part)
    return flow_state


def main():
    # Fetch the information of which flow attaches to this streamlit instance
    flow_state = _app_state_to_flow_scope(app_state, flow=os.environ["LIGHTNING_FLOW_NAME"])

    # Call the provided render function.
    # Pass it the state, scoped to the current flow.
    render_fn = _get_render_fn_from_environment()
    render_fn(flow_state)


if __name__ == "__main__":
    main()
