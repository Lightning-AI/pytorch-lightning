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

"""Utility functions for lightning Frontends."""
from __future__ import annotations

import inspect
import os
from typing import Callable

from lightning.app.core.flow import LightningFlow
from lightning.app.utilities.state import AppState


def _reduce_to_flow_scope(state: AppState, flow: str | LightningFlow) -> AppState:
    """Returns a new AppState with the scope reduced to the given flow."""
    flow_name = flow.name if isinstance(flow, LightningFlow) else flow
    flow_name_parts = flow_name.split(".")[1:]  # exclude root
    flow_state = state
    for part in flow_name_parts:
        flow_state = getattr(flow_state, part)
    return flow_state


def _get_flow_state(flow: str) -> AppState:
    """Returns an AppState scoped to the current Flow.

    Returns:
        AppState: An AppState scoped to the current Flow.
    """
    app_state = AppState()
    app_state._request_state()  # pylint: disable=protected-access
    flow_state = _reduce_to_flow_scope(app_state, flow)
    return flow_state


def _get_frontend_environment(flow: str, render_fn_or_file: Callable | str, port: int, host: str) -> os._Environ:
    """Returns an _Environ with the environment variables for serving a Frontend app set.

    Args:
        flow: The name of the flow, for example root.lit_frontend
        render_fn_or_file: A function to render
        port: The port number, for example 54321
        host: The host, for example 'localhost'

    Returns:
        os._Environ: An environment
    """
    env = os.environ.copy()
    env["LIGHTNING_FLOW_NAME"] = flow
    env["LIGHTNING_RENDER_PORT"] = str(port)
    env["LIGHTNING_RENDER_ADDRESS"] = str(host)

    if isinstance(render_fn_or_file, str):
        env["LIGHTNING_RENDER_FILE"] = render_fn_or_file
    else:
        env["LIGHTNING_RENDER_FUNCTION"] = render_fn_or_file.__name__
        env["LIGHTNING_RENDER_MODULE_FILE"] = inspect.getmodule(render_fn_or_file).__file__

    return env
