"""Utility functions for lightning Frontends."""
# Todo: Refactor stream_lit and streamlit_base to use this functionality

from __future__ import annotations

import inspect
import os
import pydoc
from typing import Callable

from lightning_app.core.flow import LightningFlow
from lightning_app.utilities.state import AppState


def get_render_fn_from_environment() -> Callable:
    """Returns the render_fn function to serve in the Frontend."""
    render_fn_name = os.environ["LIGHTNING_RENDER_FUNCTION"]
    render_fn_module_file = os.environ["LIGHTNING_RENDER_MODULE_FILE"]
    module = pydoc.importfile(render_fn_module_file)
    return getattr(module, render_fn_name)


def _reduce_to_flow_scope(state: AppState, flow: str | LightningFlow) -> AppState:
    """Returns a new AppState with the scope reduced to the given flow."""
    flow_name = flow.name if isinstance(flow, LightningFlow) else flow
    flow_name_parts = flow_name.split(".")[1:]  # exclude root
    flow_state = state
    for part in flow_name_parts:
        flow_state = getattr(flow_state, part)
    return flow_state


def get_flow_state() -> AppState:
    """Returns an AppState scoped to the current Flow.

    Returns:
        AppState: An AppState scoped to the current Flow.
    """
    app_state = AppState()
    app_state._request_state()  # pylint: disable=protected-access
    flow = os.environ["LIGHTNING_FLOW_NAME"]
    flow_state = _reduce_to_flow_scope(app_state, flow)
    return flow_state


def get_frontend_environment(flow: str, render_fn: Callable, port: int, host: str) -> os._Environ:
    """Returns an _Environ with the environment variables for serving a Frontend app set.

    Args:
        flow (str): The name of the flow, for example root.lit_frontend
        render_fn (Callable): A function to render
        port (int): The port number, for example 54321
        host (str): The host, for example 'localhost'

    Returns:
        os._Environ: An environement
    """
    env = os.environ.copy()
    env["LIGHTNING_FLOW_NAME"] = flow
    env["LIGHTNING_RENDER_FUNCTION"] = render_fn.__name__
    env["LIGHTNING_RENDER_MODULE_FILE"] = inspect.getmodule(render_fn).__file__
    env["LIGHTNING_RENDER_PORT"] = str(port)
    env["LIGHTNING_RENDER_ADDRESS"] = str(host)
    return env
