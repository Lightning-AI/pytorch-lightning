"""This file gets run by Python to lunch a Panel Server with Lightning.

From here, we will call the render_fn that the user provided to the PanelFrontend.

It requires the below environment variables to be set

- LIGHTNING_FLOW_NAME
- LIGHTNING_RENDER_ADDRESS
- LIGHTNING_RENDER_PORT

As well as either

- LIGHTNING_RENDER_FUNCTION + LIGHTNING_RENDER_MODULE_FILE

or

- LIGHTNING_RENDER_FILE


Example:

.. code-block:: bash

        python panel_serve_render_fn_or_file
"""
from __future__ import annotations

import inspect
import logging
import os

import panel as pn

from lightning_app.frontend.utilities.app_state_watcher import AppStateWatcher
from lightning_app.frontend.utilities.other import get_allowed_hosts, get_render_fn_from_environment

_logger = logging.getLogger(__name__)


def _get_render_fn():
    render_fn_name = os.environ["LIGHTNING_RENDER_FUNCTION"]
    render_fn_module_file = os.environ["LIGHTNING_RENDER_MODULE_FILE"]
    return get_render_fn_from_environment(render_fn_name, render_fn_module_file)


def _render_fn_wrapper():
    render_fn = _get_render_fn()
    app = AppStateWatcher()
    return render_fn(app)


def _get_view_fn():
    render_fn = _get_render_fn()
    if inspect.signature(render_fn).parameters:
        return _render_fn_wrapper
    return render_fn


def _get_view():
    if "LIGHTNING_RENDER_FILE" in os.environ:
        return os.environ["LIGHTNING_RENDER_FILE"]
    return _get_view_fn()


def has_panel_autoreload() -> bool:
    """Returns True if the PANEL_AUTORELOAD environment variable is set to 'yes' or 'true'.

    Please note the casing does not matter
    """
    return os.environ.get("PANEL_AUTORELOAD", "no").lower() in ["yes", "y", "true"]


def _serve():
    port = int(os.environ["LIGHTNING_RENDER_PORT"])
    address = os.environ["LIGHTNING_RENDER_ADDRESS"]
    url = os.environ["LIGHTNING_FLOW_NAME"]
    websocket_origin = get_allowed_hosts()

    # PANEL_AUTORELOAD not yet supported by Panel. See https://github.com/holoviz/panel/issues/3681
    # Todo: With lightning, the server autoreloads but the browser does not. Fix this.
    autoreload = has_panel_autoreload()

    view = _get_view()

    pn.serve(
        {url: view},
        address=address,
        port=port,
        websocket_origin=websocket_origin,
        show=False,
        autoreload=autoreload,
    )
    _logger.debug("Panel server started on port http://%s:%s/%s", address, port, url)


if __name__ == "__main__":
    _serve()
